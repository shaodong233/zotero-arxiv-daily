from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf
from tempfile import TemporaryDirectory
from dataclasses import dataclass, field
import feedparser
from tqdm import tqdm
import os
import re
from time import sleep
from loguru import logger
import requests

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
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}", agent=USER_AGENT)
        if 'Feed error for query' in feed.feed.get('title', ''):
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        new_entries = {}
        for e in feed.entries:
            if e.get("arxiv_announce_type", "new") == 'new':
                new_entries[e.id.removeprefix("oai:arXiv.org:")] = e
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
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = None
        try:
            with TemporaryDirectory() as temp_dir:
                path = os.path.join(temp_dir, "paper.pdf")
                with requests.get(pdf_url, stream=True, timeout=(10, 60), headers={"User-Agent": USER_AGENT}) as response:
                    response.raise_for_status()
                    with open(path, "wb") as file:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                file.write(chunk)
                try:
                    full_text = extract_markdown_from_pdf(path)
                except Exception as e:
                    logger.warning(f"Failed to extract full text of {title}: {e}")
        except Exception as e:
            logger.warning(f"Failed to download PDF of {title}: {e}")
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text
        )
