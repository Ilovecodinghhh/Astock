from pathlib import Path


def extract_class_source(source: str, class_name: str) -> str:
    start_marker = f"class {class_name}"
    start = source.index(start_marker)
    next_class = source.find("\nclass ", start + len(start_marker))
    if next_class == -1:
        return source[start:]
    return source[start:next_class]


def test_precomputed_relative_strength_strategy_does_not_scan_current_whitelist() -> None:
    source = Path("user_data/strategies/AlternativeStrategyExperiments.py").read_text(encoding="utf-8")

    class_source = extract_class_source(source, "FuturesPrecomputedRelativeStrengthLooseGuardStrategy")

    assert "current_whitelist" not in class_source
    assert "def _rank_percentile" not in class_source
    assert "self._rank_percentile" not in class_source
    assert "def _market_breadth" not in class_source
    assert "self._market_breadth" not in class_source


def test_precomputed_relative_strength_strategy_overrides_entry_threshold() -> None:
    source = Path("user_data/strategies/AlternativeStrategyExperiments.py").read_text(encoding="utf-8")

    class_source = extract_class_source(source, "FuturesPrecomputedRelativeStrengthLooseGuardStrategy")

    assert "universe_size = 30" in class_source
    assert "def populate_entry_trend" in class_source


def test_precomputed_relative_strength_variants_inherit_clean_strategy() -> None:
    source = Path("user_data/strategies/AlternativeStrategyExperiments.py").read_text(encoding="utf-8")

    for class_name in [
        "FuturesPrecomputedRelativeStrengthLooseGuardHighStrategy",
        "FuturesPrecomputedRelativeStrengthLooseGuardDefensiveStrategy",
        "FuturesPrecomputedRelativeStrengthLooseGuardStrictStrategy",
    ]:
        class_source = extract_class_source(source, class_name)
        assert f"class {class_name}(FuturesPrecomputedRelativeStrengthLooseGuardStrategy)" in class_source
