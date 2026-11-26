"""bench oracle grammar (ORACLES.md R13): primitives, composition,
repeat-group rate verdicts — pure, no network."""
from orchestrator import bench


def test_primitives():
    assert bench.score("contains:Kandi", "I'm Kandi.") == "PASS"
    assert bench.score("contains:kandi", "nope") == "FAIL"
    assert bench.score("regex:^\\W*yes\\W*$", "  Yes! ") == "PASS"
    assert bench.score("regex:(", "anything") == "?"        # bad pattern
    assert bench.score("one_of:cup,table", "under the TABLE") == "PASS"
    assert bench.score("json", 'here {\"a\": 1}') == "PASS"
    assert bench.score("json", "no braces") == "FAIL"
    assert bench.score("none", "whatever") == "—"


def test_not_composition():
    assert bench.score("not:contains:kandi", "nope") == "PASS"
    assert bench.score("not:contains:kandi", "I am Kandi") == "FAIL"
    assert bench.score("not:json", "plain prose") == "PASS"


def test_rate_over_repeats():
    texts = ["I'm Kandi.", "Kandi here", "who?", "Kandi.", "?"]
    assert bench.score("rate:contains:kandi/0.5-1.0", texts) == "PASS"
    assert bench.score("rate:contains:kandi/0.8-1.0", texts) == "FAIL"
    assert bench.rate_observed("rate:contains:kandi/0.5-1.0", texts) == 0.6
    # degenerate bounds still behave
    assert bench.score("rate:contains:x/0-1", texts) == "PASS"
    assert bench.score("rate:contains:x/bad", texts) == "?"


def test_score_all_uses_inner_for_rate():
    texts = ["yes", "no", "yes"]
    vs = bench.score_all("rate:contains:yes/0.5-1.0", texts)
    assert vs == ["PASS", "FAIL", "PASS"]
