from __future__ import annotations

import pandas as pd

from scripts.astock_make_pool import (
    a_share_codes_from_frame,
    codes_from_cache_dir,
    evenly_spaced_codes,
    merge_required_and_sampled_codes,
)


def test_a_share_codes_from_frame_filters_indexes_and_b_shares() -> None:
    frame = pd.DataFrame(
        {
            "code": [
                "sh.000001",
                "sh.600000",
                "sh.688001",
                "sh.900901",
                "sz.000001",
                "sz.002001",
                "sz.300001",
                "sz.200001",
                "bj.430047",
            ]
        }
    )

    assert a_share_codes_from_frame(frame) == [
        "sh.600000",
        "sh.688001",
        "sz.000001",
        "sz.002001",
        "sz.300001",
    ]


def test_evenly_spaced_codes_samples_across_sorted_code_space() -> None:
    codes = [f"sz.{index:06d}" for index in range(10)]

    assert evenly_spaced_codes(codes, 4) == ["sz.000000", "sz.000003", "sz.000006", "sz.000009"]


def test_merge_required_and_sampled_codes_keeps_required_then_adds_sample() -> None:
    all_codes = [f"sz.{index:06d}" for index in range(8)]
    required = ["sz.000006", "sz.000001"]

    merged = merge_required_and_sampled_codes(all_codes, required, limit=5)

    assert merged == ["sz.000001", "sz.000006", "sz.000000", "sz.000003", "sz.000007"]


def test_codes_from_cache_dir_reads_feather_file_names(tmp_path) -> None:
    (tmp_path / "sh.600000-d.feather").write_bytes(b"cache")
    (tmp_path / "sz.000001-d.feather").write_bytes(b"cache")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    assert codes_from_cache_dir(tmp_path) == ["sh.600000", "sz.000001"]
