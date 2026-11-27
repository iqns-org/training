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


# ---------- instrument hardening (the v2 incident lessons) ----------

def test_normalize_url_forms():
    # every shape a human has ever typed lands on the bare base
    assert bench.normalize_url("http://h:8000") == "http://h:8000"
    assert bench.normalize_url("http://h:8000/") == "http://h:8000"
    assert bench.normalize_url("http://h:8000/v1") == "http://h:8000"
    assert (bench.normalize_url("http://h:8000/v1/chat/completions")
            == "http://h:8000")


def test_lint_impossible_rate_bounds():
    # the /33-100 bug: percent-scale bounds on a fraction-scale check
    # grade FAIL forever, silently. lint names it at LOAD.
    bad = [{"id": "q1", "prompt": "p", "check": "regex:^yes$"},
           {"id": "q2", "prompt": "p",
            "check": "rate:contains:x/33-100"},
           {"id": "q3", "prompt": "p", "check": "rate:contains:x/0.8-0.2"},
           {"id": "q4", "prompt": "p", "check": "rate:contains:x/nope"},
           {"id": "q5", "prompt": "p", "check": "regex:(unclosed"}]
    errs = bench.lint_suite(bad)
    joined = " ".join(errs)
    for qid in ("q2", "q3", "q4", "q5"):
        assert qid in joined, errs
    assert "q1" not in joined                      # good ones stay silent
    assert bench.lint_suite([{"id": "ok", "prompt": "p",
                              "check": "rate:regex:.*yes|nope/0.34-1.0"}]) == []


def test_run_lints_before_any_network(tmp_path, monkeypatch):
    # a broken suite must abort the run, not grade samples against it
    def boom(*a, **k):
        raise AssertionError("run() touched the wire despite lint error")
    monkeypatch.setattr(bench, "chat_stream", boom)
    monkeypatch.setattr(bench, "_req", boom)
    cfg = {"targets": [{"name": "t", "url": "http://x", "model": "m"}],
           "questions": [{"id": "q", "prompt": "p",
                          "check": "rate:contains:x/33-100"}]}
    import pytest
    with pytest.raises(ValueError, match="lint"):
        bench.run(cfg)


def test_preflight_fails_fast(monkeypatch):
    import urllib.error
    def dead(url, **k):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(bench, "_req", dead)
    cfg = {"targets": [{"name": "t", "url": "http://dead", "model": "m"}],
           "questions": [{"id": "q", "prompt": "p", "check": "none"}]}
    import pytest
    with pytest.raises(SystemExit, match="preflight"):
        bench.run(cfg)


def test_answers_sink_keeps_full_text(monkeypatch):
    long_answer = "Y" * 300                       # beyond row truncation
    monkeypatch.setattr(bench, "_req", lambda *a, **k: {})
    monkeypatch.setattr(bench, "chat_stream",
                        lambda *a, **k: (long_answer, 1.0, 0.1, 40.0))
    cfg = {"targets": [{"name": "t", "url": "http://x", "model": "m"}],
           "questions": [{"id": "q", "prompt": "p",
                          "check": "contains:" + long_answer}],
           "repeats": 2, "preflight": False}
    sink = []
    rows = bench.run(cfg, sink=sink)
    assert rows[0]["pass"] == "PASS"
    assert len(rows[0]["answer"]) == 120          # row field truncates
    assert sink[0]["answers"] == [long_answer, long_answer]
    assert sink[0]["verdict"] == "PASS" and sink[0]["q"] == "q"
