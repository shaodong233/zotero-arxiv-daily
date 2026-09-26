from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
import feedparser
import pickle

def test_arxiv_retriever(config, monkeypatch):

    parsed_result = feedparser.parse("tests/retriever/arxiv_rss_example.xml")
    raw_parser = feedparser.parse
    def mock_feedparser_parse(url, *args, **kwargs):
        if url == f"https://rss.arxiv.org/atom/{'+'.join(config.source.arxiv.category)}":
            return parsed_result
        return raw_parser(url, *args, **kwargs)
    monkeypatch.setattr(feedparser, "parse", mock_feedparser_parse)
    
    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()
    parsed_results = [i for i in parsed_result.entries if i.get("arxiv_announce_type","new") == 'new']
    assert len(papers) == len(parsed_results)
    paper_titles = [i.title for i in papers]
    parsed_titles = [i.title for i in parsed_results]
    assert set(paper_titles) == set(parsed_titles)

RSS_FALLBACK_XML = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns="http://www.w3.org/2005/Atom" xml:lang="en-us">
  <id>http://rss.arxiv.org/atom/cs.AI</id>
  <title>cs.AI updates on arXiv.org</title>
  <entry>
    <id>oai:arXiv.org:2508.13435v1</id>
    <title>SVDformer: Direction-Aware Spectral Graph
      Embedding Learning via SVD and Transformer</title>
    <link href="https://arxiv.org/abs/2508.13435" rel="alternate" type="text/html"/>
    <summary>arXiv:2508.13435v1 Announce Type: new
Abstract: Directed graphs are widely used to model asymmetric relationships.</summary>
    <arxiv:announce_type>new</arxiv:announce_type>
    <dc:creator>Jiayu Fang, Zhiqi Shao, S T Boris Choy, Junbin Gao</dc:creator>
  </entry>
  <entry>
    <id>oai:arXiv.org:2508.13436v1</id>
    <title>Second Paper</title>
    <summary>arXiv:2508.13436v1 Announce Type: new
Abstract: Something else.</summary>
    <arxiv:announce_type>new</arxiv:announce_type>
    <dc:creator>Alice Smith and Bob Lee</dc:creator>
  </entry>
  <entry>
    <id>oai:arXiv.org:2508.13437v1</id>
    <title>Cross listed</title>
    <summary>arXiv:2508.13437v1 Announce Type: cross
Abstract: Skip me.</summary>
    <arxiv:announce_type>cross</arxiv:announce_type>
    <dc:creator>Carol</dc:creator>
  </entry>
</feed>
"""


def test_arxiv_retriever_rss_fallback(config, monkeypatch):
    import arxiv
    from zotero_arxiv_daily.retriever import arxiv_retriever as mod
    parsed_result = feedparser.parse(RSS_FALLBACK_XML)
    raw_parser = feedparser.parse
    def mock_feedparser_parse(url, *args, **kwargs):
        if url.startswith("https://rss.arxiv.org/atom/"):
            return parsed_result
        return raw_parser(url, *args, **kwargs)
    monkeypatch.setattr(feedparser, "parse", mock_feedparser_parse)

    calls = []
    def always_406(self, search, offset=0):
        calls.append(search.id_list)
        raise arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 406)
    monkeypatch.setattr(arxiv.Client, "results", always_406)
    monkeypatch.setattr(mod, "sleep", lambda *_: None)

    retriever = ArxivRetriever(config)
    raw = retriever._retrieve_raw_papers()
    assert len(calls) == 2  # 406 retried briefly, then API disabled
    assert [r.arxiv_id for r in raw] == ["2508.13435v1", "2508.13436v1"]
    assert all(isinstance(r, mod.RssRawPaper) for r in raw)
    r = raw[0]
    assert r.title == "SVDformer: Direction-Aware Spectral Graph Embedding Learning via SVD and Transformer"
    assert r.summary == "Directed graphs are widely used to model asymmetric relationships."
    assert [a.name for a in r.authors] == ["Jiayu Fang", "Zhiqi Shao", "S T Boris Choy", "Junbin Gao"]
    assert [a.name for a in raw[1].authors] == ["Alice Smith", "Bob Lee"]
    assert r.pdf_url == "https://arxiv.org/pdf/2508.13435v1"
    assert r.entry_id == "https://arxiv.org/abs/2508.13435v1"
    pickle.loads(pickle.dumps(r))

    # convert_to_paper must work on fallback items (PDF download fails gracefully)
    def offline(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(mod.requests, "get", offline)
    paper = retriever.convert_to_paper(r)
    assert paper.title == r.title and paper.abstract == r.summary
    assert paper.authors == [a.name for a in r.authors]
    assert paper.pdf_url == r.pdf_url and paper.url == r.entry_id
    assert paper.full_text is None


def test_user_agent_overrides_arxiv_default():
    from zotero_arxiv_daily.retriever import arxiv_retriever as mod
    client = mod._make_arxiv_client(num_retries=0)
    captured = {}
    def fake_send(request, **kwargs):
        captured.update(request.headers)
        raise RuntimeError("stop")
    client._session.send = fake_send
    try:
        client._session.get("https://export.arxiv.org/api/query", headers={"user-agent": "arxiv.py/2.3.2"})
    except RuntimeError:
        pass
    assert captured["User-Agent"] == mod.USER_AGENT


def test_arxiv_live_api_with_user_agent():
    """Live check of the export API with our User-Agent (skips, not fails, on HTTP errors)."""
    import warnings
    import arxiv
    import pytest
    from zotero_arxiv_daily.retriever import arxiv_retriever as mod
    client = mod._make_arxiv_client(num_retries=1, delay_seconds=3, page_size=5)
    try:
        results = list(client.results(arxiv.Search(id_list=["2508.13435v1"])))
    except arxiv.HTTPError as exc:
        warnings.warn(f"arXiv live API check: HTTP {exc.status}")
        pytest.skip(f"arXiv API HTTP {exc.status}")
    assert len(results) == 1
