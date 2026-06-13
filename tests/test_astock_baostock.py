from scripts.astock_baostock import iter_codes, normalize_baostock_code, should_skip_download


def test_normalize_baostock_code_adds_exchange_prefix() -> None:
    assert normalize_baostock_code("600000") == "sh.600000"
    assert normalize_baostock_code("000001") == "sz.000001"
    assert normalize_baostock_code("sh.600519") == "sh.600519"


def test_should_skip_download_when_cache_exists(tmp_path) -> None:
    cache = tmp_path / "sh.600000-d.feather"
    cache.write_bytes(b"cache")

    assert should_skip_download(tmp_path, "600000", "d") is True
    assert should_skip_download(tmp_path, "000001", "d") is False


def test_iter_codes_combines_cli_and_pool_file(tmp_path) -> None:
    pool = tmp_path / "pool.txt"
    pool.write_text("600000\n# comment\n000001\n", encoding="utf-8")

    assert iter_codes(["600519"], pool) == ["600519", "600000", "000001"]
