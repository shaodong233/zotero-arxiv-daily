from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
from zotero_arxiv_daily.retriever import arxiv_retriever as arxiv_mod
import feedparser
import pickle
from pathlib import Path


def _xml_response(content: bytes, status_code: int = 200):
    class Resp:
        def __init__(self):
            self.status_code = status_code
            self.content = content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise arxiv_mod.requests.HTTPError(f"HTTP {self.status_code}")

    return Resp()


def _mock_rss_get(monkeypatch, content: bytes, status_code: int = 200):
    def fake_get(url, *args, **kwargs):
        if "rss.arxiv.org/atom/" in url:
            return _xml_response(content, status_code)
        raise AssertionError(f"unexpected GET {url}")
    monkeypatch.setattr(arxiv_mod.requests, "get", fake_get)
    monkeypatch.setattr(arxiv_mod, "sleep", lambda *_: None)


def test_arxiv_retriever(config, monkeypatch):
    xml = Path("tests/retriever/arxiv_rss_example.xml").read_bytes()
    parsed_result = feedparser.parse(xml)
    _mock_rss_get(monkeypatch, xml)

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
    _mock_rss_get(monkeypatch, RSS_FALLBACK_XML.encode("utf-8"))

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


EMPTY_FEED_XML = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">
  <title>cs.AI updates on arXiv.org</title>
  <updated>2026-11-22T05:00:00+00:00</updated>
</feed>
"""


def test_load_rss_retries_empty_feed_on_weekday(monkeypatch):
    payloads = [EMPTY_FEED_XML.encode("utf-8"), RSS_FALLBACK_XML.encode("utf-8")]
    sleeps = []

    def fake_get(url, *args, **kwargs):
        return _xml_response(payloads.pop(0) if payloads else RSS_FALLBACK_XML.encode("utf-8"))

    monkeypatch.setattr(arxiv_mod.requests, "get", fake_get)
    monkeypatch.setattr(arxiv_mod, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        arxiv_mod, "_is_expected_empty_rss_day", lambda now=None: False
    )
    entries = arxiv_mod._load_new_arxiv_entries("https://rss.arxiv.org/atom/cs.AI")
    assert set(entries) == {"2508.13435v1", "2508.13436v1"}
    assert sleeps == [60]


def test_load_rss_does_not_retry_empty_weekend_feed(monkeypatch):
    sleeps = []

    def fake_get(url, *args, **kwargs):
        return _xml_response(EMPTY_FEED_XML.encode("utf-8"))

    monkeypatch.setattr(arxiv_mod.requests, "get", fake_get)
    monkeypatch.setattr(arxiv_mod, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(arxiv_mod, "_is_expected_empty_rss_day", lambda now=None: True)
    entries = arxiv_mod._load_new_arxiv_entries("https://rss.arxiv.org/atom/cs.AI")
    assert entries == {}
    assert sleeps == []


def test_load_rss_retries_http_406_then_raises(monkeypatch):
    calls = []

    def fake_get(url, *args, **kwargs):
        calls.append(url)
        return _xml_response(b"", status_code=406)

    monkeypatch.setattr(arxiv_mod.requests, "get", fake_get)
    monkeypatch.setattr(arxiv_mod, "sleep", lambda *_: None)
    monkeypatch.setattr(arxiv_mod, "_is_expected_empty_rss_day", lambda now=None: False)
    try:
        arxiv_mod._load_new_arxiv_entries("https://rss.arxiv.org/atom/cs.AI")
        raise AssertionError("expected RssFetchError")
    except arxiv_mod.RssFetchError as exc:
        assert exc.status == 406
    assert len(calls) == 4


def test_expected_empty_rss_day_weekend_vs_weekday():
    from datetime import datetime
    saturday = datetime(2026, 11, 21, 2, 0, tzinfo=arxiv_mod.EASTERN)
    monday = datetime(2026, 11, 23, 2, 0, tzinfo=arxiv_mod.EASTERN)
    assert arxiv_mod._is_expected_empty_rss_day(saturday) is True
    assert arxiv_mod._is_expected_empty_rss_day(monday) is False



LIST_NEW_HTML = """
<html><body>
<h3>New submissions (showing 1 of 1 entries)</h3>
<dt>
  <a href="/abs/2611.31284" title="Abstract" id="2611.31284">arXiv:2611.31284</a>
</dt>
<dd>
  <div class="meta">
    <div class='list-title mathjax'><span class='descriptor'>Title:</span> Hello World Paper</div>
    <div class='list-authors'><a href="https://arxiv.org/search/?query=Smith">Alice Smith</a></div>
    <p class='mathjax'>An abstract here.</p>
  </div>
</dd>
<h3>Cross submissions (showing 1 of 1 entries)</h3>
<dt>
  <a href="/abs/2611.99999" title="Abstract">arXiv:2611.99999</a>
</dt>
<dd>
  <div class='list-title mathjax'><span class='descriptor'>Title:</span> Should Skip</div>
  <div class='list-authors'>Bob</div>
  <p class='mathjax'>Cross.</p>
</dd>
</body></html>
"""


def test_parse_new_list_html_skips_cross_and_replacements():
    entries = arxiv_mod._parse_new_list_html(LIST_NEW_HTML)
    assert set(entries) == {"2611.31284"}
    entry = entries["2611.31284"]
    assert entry.get("title") == "Hello World Paper"
    assert entry.get("summary") == "An abstract here."
    assert entry.get("author") == "Alice Smith"
    raw = arxiv_mod._rss_entry_to_raw("2611.31284", entry)
    assert raw.title == "Hello World Paper"
    assert raw.summary == "An abstract here."
    assert [a.name for a in raw.authors] == ["Alice Smith"]


def test_load_rss_falls_back_to_new_list_on_empty_weekday(monkeypatch):
    sleeps = []

    def fake_get(url, *args, **kwargs):
        if "rss.arxiv.org" in url:
            return _xml_response(EMPTY_FEED_XML.encode("utf-8"))
        if url.endswith("/list/cs.AI/new"):
            class Resp:
                status_code = 200
                text = LIST_NEW_HTML
                def raise_for_status(self):
                    return None
            return Resp()
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(arxiv_mod.requests, "get", fake_get)
    monkeypatch.setattr(arxiv_mod, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(arxiv_mod, "_is_expected_empty_rss_day", lambda now=None: False)
    entries = arxiv_mod._load_new_arxiv_entries(
        "https://rss.arxiv.org/atom/cs.AI", categories=["cs.AI"]
    )
    assert set(entries) == {"2611.31284"}
    assert sleeps == []
