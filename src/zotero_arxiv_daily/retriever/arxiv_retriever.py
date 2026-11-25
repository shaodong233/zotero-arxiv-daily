from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf
from tempfile import TemporaryDirectory
from dataclasses import dataclass, field
import feedparser
from tqdm import tqdm
import multiprocessing
import os
import re
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
from loguru import logger
import requests

T = TypeVar("T")
DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 120

# arXiv asks API clients to identify themselves. Since ~2026-09-23 the export API
# answers HTTP 406 to requests from GitHub Actions that use a generic User-Agent.
USER_AGENT = (
    "zotero-arxiv-daily/1.0 "
    "(+https://github.com/shaodong233/zotero-arxiv-daily; mailto:shaodongking@gmail.com)"
)
# 403/406 have been observed as (soft) blocks from arXiv on shared CI IPs.
RETRIABLE_STATUS = (403, 406, 429, 503)
# Blocks (403/406) rarely clear within minutes; retry briefly, then fall back to RSS.
BLOCK_STATUS = (403, 406)
# RSS is rebuilt at midnight Eastern; retry empty/blocked fetches before giving up.
RSS_MAX_ATTEMPTS = 4
RSS_RETRY_SECONDS = (60, 180, 300)
EASTERN = ZoneInfo("America/New_York")


class _UserAgentSession(requests.Session):
    """requests.Session that always sends USER_AGENT.

    arxiv==2.4.0 passes ``headers={"user-agent": "arxiv.py/2.3.2"}`` on every request,
    which would override session-level headers, so we replace it at request time.
    """

    def request(self, method, url, *args, headers=None, **kwargs):
        merged = {k: v for k, v in (headers or {}).items() if k.lower() != "user-agent"}
        merged["User-Agent"] = USER_AGENT
        return super().request(method, url, *args, headers=merged, **kwargs)


def _make_arxiv_client(**kwargs) -> arxiv.Client:
    client = arxiv.Client(**kwargs)
    if hasattr(client, "_session"):
        session = _UserAgentSession()
        session.headers["User-Agent"] = USER_AGENT
        client._session = session
    else:  # pragma: no cover - future arxiv versions
        logger.warning("arxiv.Client has no _session; custom User-Agent not applied")
    return client


@dataclass
class RssAuthor:
    name: str


@dataclass
class RssRawPaper:
    """Minimal stand-in for arxiv.Result built from an RSS entry.

    Exposes the attributes used by ArxivRetriever.convert_to_paper.
    """
    arxiv_id: str
    title: str
    summary: str
    authors: list = field(default_factory=list)

    @property
    def entry_id(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def pdf_url(self) -> str:
        return f"https://arxiv.org/pdf/{self.arxiv_id}"


def _rss_entry_to_raw(arxiv_id: str, entry) -> RssRawPaper:
    title = re.sub(r"\s+", " ", entry.get("title", "")).strip()
    summary = entry.get("summary", "") or ""
    # RSS summary looks like "arXiv:XXXX Announce Type: new \nAbstract: ..."
    m = re.search(r"Abstract:\s*(.*)", summary, flags=re.S)
    abstract = (m.group(1) if m else summary).strip()
    creator = entry.get("author") or ""
    if not creator and entry.get("authors"):
        creator = ", ".join(a.get("name", "") for a in entry["authors"])
    authors = [RssAuthor(n.strip()) for n in re.split(r",\s*|\s+and\s+", creator) if n.strip()]
    return RssRawPaper(arxiv_id=arxiv_id, title=title, summary=abstract, authors=authors)


def _strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def _short_id(result) -> str:
    try:
        return result.get_short_id()
    except Exception:
        return str(getattr(result, "entry_id", "")).rsplit("/", 1)[-1]


class RssFetchError(Exception):
    """HTTP failure when downloading an arXiv Atom/RSS feed."""

    def __init__(self, status: int, url: str, message: str | None = None):
        self.status = status
        self.url = url
        super().__init__(message or f"HTTP {status} fetching {url}")


def _is_expected_empty_rss_day(now: datetime | None = None) -> bool:
    """arXiv does not announce Friday or Saturday, so Sat/Sun ET feeds are empty."""
    now = now or datetime.now(EASTERN)
    return now.weekday() >= 5  # Saturday=5, Sunday=6


def _fetch_atom_feed(url: str) -> tuple[Any, int]:
    """Download an arXiv Atom feed with our User-Agent and parse it.

    feedparser.parse(url) can swallow HTTP 403/406 into an empty feed; fetch
    with requests first so callers can retry or fail visibly.
    """
    response = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
    if response.status_code in RETRIABLE_STATUS:
        raise RssFetchError(response.status_code, url)
    response.raise_for_status()
    feed = feedparser.parse(response.content, agent=USER_AGENT)
    return feed, response.status_code


def _new_entries_from_feed(feed) -> dict:
    if "Feed error for query" in feed.feed.get("title", ""):
        raise Exception(f"Invalid ARXIV_QUERY: {feed.feed.get('title', '')}.")
    new_entries = {}
    for e in feed.entries:
        if e.get("arxiv_announce_type", "new") == "new":
            new_entries[e.id.removeprefix("oai:arXiv.org:")] = e
    return new_entries


def _log_rss_feed(url: str, feed, status: int, new_count: int) -> None:
    types = Counter(e.get("arxiv_announce_type", "missing") for e in feed.entries)
    logger.info(
        f"arXiv RSS {url} HTTP {status} updated={feed.feed.get('updated', 'unknown')} "
        f"entries={len(feed.entries)} types={dict(types)} new={new_count}"
    )



_NEW_SECTION_RE = re.compile(
    r"<h3>New submissions.*?</h3>(.*?)(?:<h3>(?:Cross|Replacement) submissions|$)",
    flags=re.I | re.S,
)
_DT_DD_RE = re.compile(r"<dt>(.*?)</dt>\s*<dd>(.*?)</dd>", flags=re.I | re.S)
_ABS_ID_RE = re.compile(r'href\s*=\s*"/abs/([^"\s]+)"', flags=re.I)
_TITLE_RE = re.compile(
    r"class=['\"]list-title[^'\"]*['\"][^>]*>.*?</span>\s*(.*?)</div>",
    flags=re.I | re.S,
)
_AUTHORS_RE = re.compile(r"class=['\"]list-authors['\"][^>]*>(.*?)</div>", flags=re.I | re.S)
_ABSTRACT_RE = re.compile(r"<p class=['\"]mathjax['\"]>(.*?)</p>", flags=re.I | re.S)


class HtmlListEntry:
    """feedparser-like entry built from arXiv /list/{cat}/new HTML."""

    def __init__(self, arxiv_id: str, title: str, abstract: str, authors: str):
        self.id = f"oai:arXiv.org:{arxiv_id}"
        self._data = {
            "title": title,
            "summary": abstract,
            "author": authors,
            "arxiv_announce_type": "new",
        }

    def get(self, key, default=None):
        return self._data.get(key, default)


def _strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .strip()
    )


def _parse_new_list_html(html: str) -> dict:
    """Parse 'New submissions' only (not cross-lists or replacements)."""
    match = _NEW_SECTION_RE.search(html)
    if not match:
        return {}
    entries = {}
    for dt, dd in _DT_DD_RE.findall(match.group(1)):
        id_match = _ABS_ID_RE.search(dt)
        if not id_match:
            continue
        arxiv_id = id_match.group(1)
        title_m = _TITLE_RE.search(dd)
        authors_m = _AUTHORS_RE.search(dd)
        abstract_m = _ABSTRACT_RE.search(dd)
        title = _strip_html(title_m.group(1)) if title_m else arxiv_id
        authors = _strip_html(authors_m.group(1)) if authors_m else ""
        abstract = _strip_html(abstract_m.group(1)) if abstract_m else ""
        entries[arxiv_id] = HtmlListEntry(arxiv_id, title, abstract, authors)
    return entries


def _fetch_new_list_entries(categories: list[str]) -> dict:
    """Fallback: HTML new listings, which follow the 20:00 ET announcement."""
    merged: dict = {}
    for cat in categories:
        url = f"https://arxiv.org/list/{cat}/new"
        try:
            response = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
            if response.status_code in RETRIABLE_STATUS:
                logger.warning(f"arXiv /new list HTTP {response.status_code} for {url}")
                continue
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(f"arXiv /new list request failed for {url}: {exc!r}")
            continue
        parsed = _parse_new_list_html(response.text)
        logger.info(f"arXiv /new list {url} HTTP {response.status_code} new={len(parsed)}")
        merged.update(parsed)
    return merged


def _load_new_arxiv_entries(url: str, categories: list[str] | None = None) -> dict:
    """Fetch the daily Atom feed, retrying transient HTTP errors and empty weekday feeds.

    arXiv RSS is rebuilt at midnight Eastern, several hours after the 20:00 ET
    announcement. When a weekday RSS fetch is empty, fall back to
    /list/{category}/new, which tracks the announcement mailing.
    """
    new_entries: dict = {}
    last_error: Exception | None = None
    expected_empty = _is_expected_empty_rss_day()
    attempts = 1 if expected_empty else RSS_MAX_ATTEMPTS
    categories = categories or []
    for attempt in range(attempts):
        try:
            feed, status = _fetch_atom_feed(url)
            last_error = None
        except RssFetchError as exc:
            last_error = exc
            logger.warning(f"{exc}; attempt {attempt + 1}/{attempts}")
            if not expected_empty and categories:
                fallback = _fetch_new_list_entries(categories)
                if fallback:
                    logger.warning(
                        f"Using {len(fallback)} papers from arXiv /new listings after RSS HTTP {exc.status}"
                    )
                    return fallback
            if attempt < attempts - 1:
                sleep(RSS_RETRY_SECONDS[min(attempt, len(RSS_RETRY_SECONDS) - 1)])
                continue
            break
        except requests.RequestException as exc:
            last_error = exc
            logger.warning(f"arXiv RSS request failed ({exc!r}); attempt {attempt + 1}/{attempts}")
            if not expected_empty and categories:
                fallback = _fetch_new_list_entries(categories)
                if fallback:
                    logger.warning(
                        f"Using {len(fallback)} papers from arXiv /new listings after RSS request error"
                    )
                    return fallback
            if attempt < attempts - 1:
                sleep(RSS_RETRY_SECONDS[min(attempt, len(RSS_RETRY_SECONDS) - 1)])
                continue
            break
        new_entries = _new_entries_from_feed(feed)
        _log_rss_feed(url, feed, status, len(new_entries))
        if new_entries or expected_empty:
            break
        if not expected_empty and categories:
            fallback = _fetch_new_list_entries(categories)
            if fallback:
                logger.warning(
                    f"arXiv RSS empty; using {len(fallback)} papers from /new listings "
                    "(RSS rebuilds at midnight ET, after the 20:00 ET announcement)"
                )
                return fallback
        if attempt < attempts - 1:
            wait = RSS_RETRY_SECONDS[min(attempt, len(RSS_RETRY_SECONDS) - 1)]
            logger.warning(
                f"arXiv RSS returned 0 new papers before the midnight-ET rebuild "
                f"may be ready; retry {attempt + 1}/{attempts} in {wait}s"
            )
            sleep(wait)
    if last_error and not new_entries:
        raise last_error
    if not new_entries:
        if expected_empty:
            logger.info(
                "arXiv RSS has no new papers (expected on Saturday/Sunday ET and arXiv holidays)"
            )
        else:
            logger.warning(
                "arXiv RSS still has 0 new papers after retries on a weekday; "
                "no email will be sent if every source is empty"
            )
    return new_entries


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list:
        # Prefer fewer inner retries; outer loop handles 403/406/429/503 with backoff.
        client = _make_arxiv_client(num_retries=3, delay_seconds=15, page_size=50)
        query = '+'.join(self.config.source.arxiv.category)
        # Daily Atom feed, rebuilt at midnight Eastern. See _load_new_arxiv_entries.
        new_entries = _load_new_arxiv_entries(
            f"https://rss.arxiv.org/atom/{query}",
            categories=list(self.config.source.arxiv.category),
        )
        all_paper_ids = list(new_entries.keys())
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        def rss_fallback(ids: list[str]) -> list[RssRawPaper]:
            return [_rss_entry_to_raw(pid, new_entries[pid]) for pid in ids]

        # Get full information of each paper from arxiv api
        batch_size = 5
        max_batch_retries = 8
        max_block_retries = 2
        raw_papers = []
        api_disabled = False
        fallback_count = 0
        bar = tqdm(total=len(all_paper_ids))
        # Cool down after RSS / previous runs from shared Actions IPs
        if all_paper_ids:
            sleep(5)
        for i in range(0, len(all_paper_ids), batch_size):
            ids = all_paper_ids[i:i + batch_size]
            batch = None
            if not api_disabled:
                search = arxiv.Search(id_list=ids)
                for attempt in range(max_batch_retries):
                    try:
                        batch = list(client.results(search))
                        break
                    except arxiv.HTTPError as exc:
                        is_block = exc.status in BLOCK_STATUS
                        limit = max_block_retries if is_block else max_batch_retries
                        if exc.status in RETRIABLE_STATUS and attempt < limit - 1:
                            base = 30 if is_block else 45
                            wait = min(300, base * (2 ** attempt))
                            logger.warning(
                                f"arXiv API {exc.status} on batch {i // batch_size}, "
                                f"retry {attempt + 1}/{limit} in {wait}s"
                            )
                            sleep(wait)
                            continue
                        logger.warning(
                            f"arXiv API failed on batch {i // batch_size} ({exc}); "
                            "falling back to RSS metadata for the remaining papers"
                        )
                        api_disabled = True
                        break
                    except Exception as exc:
                        logger.warning(
                            f"arXiv API error on batch {i // batch_size} ({exc!r}); "
                            "falling back to RSS metadata for the remaining papers"
                        )
                        api_disabled = True
                        break
            if batch is None:
                batch = rss_fallback(ids)
                fallback_count += len(batch)
            else:
                got = {_strip_version(_short_id(r)) for r in batch}
                missing = [pid for pid in ids if _strip_version(pid) not in got]
                if missing:
                    logger.warning(f"arXiv API missed {missing}; using RSS metadata for them")
                    batch.extend(rss_fallback(missing))
                    fallback_count += len(missing)
            bar.update(len(batch))
            raw_papers.extend(batch)
            if not api_disabled and i + batch_size < len(all_paper_ids):
                sleep(10)
        bar.close()
        if fallback_count:
            logger.warning(
                f"Used RSS fallback metadata for {fallback_count}/{len(raw_papers)} arXiv papers"
            )
        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        # Metadata only here. PDF full-text is fetched later for the top-N
        # reranked papers so daily Actions do not download hundreds of PDFs.
        return Paper(
            source=self.name,
            title=raw_paper.title,
            authors=[a.name for a in raw_paper.authors],
            abstract=raw_paper.summary,
            url=raw_paper.entry_id,
            pdf_url=raw_paper.pdf_url,
            full_text=None,
        )


def _download_pdf(pdf_url: str, path: str) -> None:
    with requests.get(
        pdf_url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers={"User-Agent": USER_AGENT}
    ) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_pdf(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    """Kill hung pymupdf/network work after timeout (thread timeouts cannot)."""
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()
    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()
    if status == "ok":
        return payload
    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def enrich_paper_full_text(paper: Paper, timeout: float = PDF_EXTRACT_TIMEOUT) -> None:
    """Download+extract PDF full text into paper.full_text (best-effort)."""
    if paper.full_text or not paper.pdf_url:
        return
    paper.full_text = _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=timeout,
        operation="PDF extraction",
        paper_title=paper.title,
    )
