from pathlib import Path

from scripts.astock_batch_download import pending_codes


def test_pending_codes_skips_existing_cache(tmp_path: Path) -> None:
    pool = ["600000", "000001", "600519"]
    (tmp_path / "sh.600000-d.feather").write_bytes(b"cache")

    assert pending_codes(pool, tmp_path, "d") == ["000001", "600519"]
