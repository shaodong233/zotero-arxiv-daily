from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import os
from time import sleep
from loguru import logger
import requests


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        # Prefer fewer inner retries; outer loop handles 429/503 with longer backoff.
        client = arxiv.Client(num_retries=3, delay_seconds=15, page_size=50)
        query = '+'.join(self.config.source.arxiv.category)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        all_paper_ids = [i.id.removeprefix("oai:arXiv.org:") for i in feed.entries if i.get("arxiv_announce_type","new") == 'new']
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Get full information of each paper from arxiv api
        batch_size = 5
        max_batch_retries = 8
        bar = tqdm(total=len(all_paper_ids))
        # Cool down after RSS / previous runs from shared Actions IPs
        sleep(5)
        for i in range(0, len(all_paper_ids), batch_size):
            search = arxiv.Search(id_list=all_paper_ids[i:i + batch_size])
            for attempt in range(max_batch_retries):
                try:
                    batch = list(client.results(search))
                    bar.update(len(batch))
                    raw_papers.extend(batch)
                    break
                except arxiv.HTTPError as exc:
                    retriable = exc.status in (429, 503)
                    if retriable and attempt < max_batch_retries - 1:
                        wait = min(300, 45 * (2 ** attempt))
                        logger.warning(
                            f"arXiv API {exc.status} on batch {i // batch_size}, "
                            f"retry {attempt + 1}/{max_batch_retries} in {wait}s"
                        )
                        sleep(wait)
                    else:
                        raise
            if i + batch_size < len(all_paper_ids):
                sleep(10)
        bar.close()

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
                with requests.get(pdf_url, stream=True, timeout=(10, 60)) as response:
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
