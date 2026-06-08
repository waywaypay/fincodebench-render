"""Report-methodology tests (Phase 4).

The report must make scoring-scale differences explicit instead of hiding them:
binary methods (functional/fuzzy_number/exact_json) can only score 0 or 1, while
partial-credit methods (fuzzy_dict/llm_judge) are continuous 0-1. Comparing a
binary category's mean against a partial-credit one overstates the gap, so the
report flags it — and surfaces the diagnostic failure breakdown so a low binary
score can be read as "benchmark-fault" vs "real capability gap".
"""
import eval as e


def _result(task_id, category, method, score, normalized=None, failure_type=None):
    sr = {"score": score, "method": method}
    if normalized is not None:
        sr["normalized"] = normalized
    if failure_type is not None:
        sr["diagnostic_judge_result"] = {"failure_type": failure_type, "method": "diagnostic_judge"}
    return {"task_id": task_id, "category": category, "difficulty": "medium", "score_result": sr}


def test_scale_taxonomy_classifies_methods():
    assert e._scale_of("functional") == "binary"
    assert e._scale_of("fuzzy_number") == "binary"
    assert e._scale_of("exact_json") == "binary"
    assert e._scale_of("fuzzy_dict") == "partial_credit"
    assert e._scale_of("llm_judge") == "partial_credit"
    assert e._scale_of("mystery") == "unknown"


def test_cross_category_warning_fires_when_scales_differ():
    results = [
        _result("codegen-1", "code_generation", "functional", 1.0),
        _result("agentic-1", "agentic", "llm_judge", 3, normalized=0.75),
    ]
    tasks = {r["task_id"]: {"id": r["task_id"]} for r in results}
    rep = e.generate_report(results, tasks)
    m = rep["scoring_methodology"]
    assert m["cross_category_comparison_warning"] is True
    assert m["binary_categories"] == ["code_generation"]
    assert m["partial_credit_categories"] == ["agentic"]
    assert m["detail"]   # human-readable explanation present


def test_no_cross_warning_when_all_one_scale():
    results = [
        _result("codegen-1", "code_generation", "functional", 1.0),
        _result("extract-1", "extraction", "exact_json", 1.0),
    ]
    tasks = {r["task_id"]: {"id": r["task_id"]} for r in results}
    rep = e.generate_report(results, tasks)
    assert rep["scoring_methodology"]["cross_category_comparison_warning"] is False
    assert rep["scoring_methodology"]["detail"] is None


def test_within_category_mixed_scoring_flag():
    # computation mixes fuzzy_number (binary) and fuzzy_dict (partial credit).
    results = [
        _result("comp-num", "computation", "fuzzy_number", 1.0),
        _result("comp-dict", "computation", "fuzzy_dict", 0.5),
    ]
    tasks = {r["task_id"]: {"id": r["task_id"]} for r in results}
    rep = e.generate_report(results, tasks)
    assert rep["by_category"]["computation"]["mixed_scoring_warning"] is True
    assert set(rep["by_category"]["computation"]["scales"]) == {"binary", "partial_credit"}


def test_functional_failure_breakdown_is_aggregated():
    results = [
        _result("cg-1", "code_generation", "functional", 0.0, failure_type="signature_mismatch"),
        _result("cg-2", "code_generation", "functional", 0.0, failure_type="signature_mismatch"),
        _result("cg-3", "code_generation", "functional", 0.0, failure_type="wrong_formula"),
        _result("cg-4", "code_generation", "functional", 1.0),  # passed, no diagnosis
    ]
    tasks = {r["task_id"]: {"id": r["task_id"]} for r in results}
    rep = e.generate_report(results, tasks)
    breakdown = rep["scoring_methodology"]["functional_failure_breakdown"]
    assert breakdown == {"signature_mismatch": 2, "wrong_formula": 1}


def test_report_carries_benchmark_version():
    rep = e.generate_report([], {})
    assert rep["benchmark_version"] == e.BENCHMARK_VERSION


def test_tool_loop_guard_is_shared_by_cli_eval():
    tasks = {"codegen-1": {"id": "codegen-1", "scoring_type": "functional"}}
    results = [{"task_id": "codegen-1", "trajectory": [{"blocks": [{"type": "text", "text": "no tools"}]}]}]
    try:
        e.ensure_tools_actually_ran(results, tasks, "venice", "text-only")
    except RuntimeError as err:
        assert "No tool executions were recorded" in str(err)
        assert "venice/text-only" in str(err)
    else:
        raise AssertionError("expected RuntimeError")


def test_tool_loop_guard_allows_toolless_non_tool_tasks():
    tasks = {"extract-1": {"id": "extract-1", "scoring_type": "exact_json"}}
    results = [{"task_id": "extract-1", "trajectory": [{"blocks": [{"type": "text", "text": "{}"}]}]}]
    e.ensure_tools_actually_ran(results, tasks)
