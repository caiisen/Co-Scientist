from __future__ import annotations

from co_scientist.search.fusion import _result_key, fuse_and_deduplicate
from co_scientist.search.platforms.base import SearchResult


def _make(
    title: str, doi: str | None = None, source: str = "pubmed", url: str | None = None
) -> SearchResult:
    return SearchResult(title=title, abstract="", year=None, source=source, url=url, doi=doi)


class TestResultKey:
    def test_doi_normalized(self):
        r = _make("Title", doi="https://doi.org/10.1000/xyz")
        assert _result_key(r) == "doi:10.1000/xyz"

    def test_doi_plain(self):
        r = _make("Title", doi="10.1000/xyz")
        assert _result_key(r) == "doi:10.1000/xyz"

    def test_arxiv_url_key(self):
        r = _make("Title", source="arxiv", url="http://arxiv.org/abs/2301.12345v2")
        key = _result_key(r)
        assert "v2" not in key
        assert "arxiv" in key or "2301.12345" in key

    def test_title_fallback(self):
        r = _make("My Research Paper!")
        key = _result_key(r)
        assert key.startswith("title:")
        assert "!" not in key


class TestFuseAndDeduplicate:
    def test_empty(self):
        assert fuse_and_deduplicate([]) == []

    def test_single_list_preserves_order(self):
        a = _make("A", doi="10/a")
        b = _make("B", doi="10/b")
        c = _make("C", doi="10/c")
        result = fuse_and_deduplicate([[a, b, c]])
        assert [r.doi for r in result] == ["10/a", "10/b", "10/c"]

    def test_deduplicates_same_doi(self):
        a1 = _make("A v1", doi="10/a", source="pubmed")
        a2 = _make("A v2", doi="10/a", source="semantic_scholar")
        b = _make("B", doi="10/b")
        result = fuse_and_deduplicate([[a1, b], [a2]])
        dois = [r.doi for r in result]
        assert dois.count("10/a") == 1

    def test_rrf_promotes_multi_list_hits(self):
        # Paper X appears in both lists (rank 0 each) → higher score than paper Y (only in one)
        x1 = _make("X first", doi="10/x")
        x2 = _make("X second", doi="10/x")
        y = _make("Y", doi="10/y")
        z = _make("Z", doi="10/z")
        result = fuse_and_deduplicate([[x1, y], [x2, z]])
        # X should be first since it appears in both lists at rank 0
        assert result[0].doi == "10/x"

    def test_no_doi_title_dedup(self):
        r1 = _make("Deep Learning Advances", source="arxiv")
        r2 = _make("Deep Learning Advances", source="semantic_scholar")
        result = fuse_and_deduplicate([[r1], [r2]])
        assert len(result) == 1
