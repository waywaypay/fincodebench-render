"""Ground-truth integrity tests.

A benchmark is only as trustworthy as its expected outputs. These tests
recompute every `computation` task's ground truth from the formula stated in the
task's own prompt and assert it matches what is stored in tasks.json — so a
hand-transcription slip can never silently invert the signal again. They also
verify, end to end through the real scorer, that a correct solver scores 1.0 and
that the debug-002 functional test actually checks the regression output (not
just the bookkeeping).

All references here are pure-stdlib so the suite runs with no third-party deps.
"""
import json
from pathlib import Path

import scorer

TASKS = {
    t["id"]: t
    for t in json.loads(
        (Path(__file__).resolve().parents[1] / "tasks" / "tasks.json").read_text()
    )
}


# ── Reference implementations (each mirrors its prompt's stated formula) ────────
def _ufcf():
    rows = {"2021": (1200, 340, 450, 85), "2022": (1450, 365, 510, 120), "2023": (1680, 390, 480, 95)}
    return {y: round(e * (1 - 0.21) + da - cx - wc, 2) for y, (e, da, cx, wc) in rows.items()}


def _altman_z():
    CA, CL, TA, RE, EBIT, MC, TL, REV = 2450, 1230, 8900, 3200, 1100, 12500, 4200, 9800
    X1, X2, X3, X4, X5 = (CA - CL) / TA, RE / TA, EBIT / TA, MC / TL, REV / TA
    return round(1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + 1.0 * X5, 3)


def _iroic():
    d = {"2020": (800, 4500), "2021": (950, 5200), "2022": (1100, 5800), "2023": (1320, 6400), "2024": (1580, 7100)}
    ys = sorted(d)
    out = {ys[0]: None}
    for i in range(1, len(ys)):
        e, ic = d[ys[i]]
        pe, pic = d[ys[i - 1]]
        out[ys[i]] = round((e - pe) * (1 - 0.21) / (ic - pic), 4)
    return out


def _wacc():
    E, D, Rf, Beta, ERP, Rd, T = 15400, 4200, 0.042, 1.35, 0.055, 0.068, 0.21
    V = E + D
    Re = Rf + Beta * ERP
    return round((E / V) * Re + (D / V) * Rd * (1 - T), 6)


def _gross_margin():
    rows = {"Q1 2024": (5234, 2891), "Q2 2024": (5567, 3012), "Q3 2024": (5891, 3156), "Q4 2024": (6234, 3287)}
    return {k: round((r - c) / r, 4) for k, (r, c) in rows.items()}


def _beat_rate():
    rows = [(1000, 1100, 1145), (1100, 1200, 1089), (1150, 1250, 1267), (1200, 1350, 1389),
            (1300, 1400, 1298), (1350, 1450, 1478), (1400, 1500, 1523), (1500, 1600, 1612)]
    return round(sum(1 for lo, hi, a in rows if a > (lo + hi) / 2) / len(rows), 4)


# ── Stored ground truth must equal the recomputed value (the four fixed tasks) ──
def test_computation_002_ufcf_matches_formula():
    assert TASKS["computation-002"]["expected_output"] == _ufcf()


def test_computation_003_altman_matches_formula():
    assert TASKS["computation-003"]["expected_output"] == _altman_z()


def test_computation_005_iroic_matches_formula():
    assert TASKS["computation-005"]["expected_output"] == _iroic()


def test_computation_006_wacc_matches_formula():
    assert TASKS["computation-006"]["expected_output"] == _wacc()


# ── End to end: a correct solver must score 1.0 through the real scorer ─────────
def _score(task_id, response):
    return scorer.score_task(TASKS[task_id], {"final_response": response})["score"]


def test_correct_solver_passes_every_computation_task():
    # The invariant the four bugs violated: a model that follows the prompt passes.
    assert _score("computation-001", json.dumps(_gross_margin())) == 1.0
    assert _score("computation-002", json.dumps(_ufcf())) == 1.0
    assert _score("computation-003", f"The Altman Z-score is {_altman_z()}.") == 1.0
    assert _score("computation-004", f"Beat rate = {_beat_rate()}") == 1.0
    assert _score("computation-005", json.dumps(_iroic())) == 1.0
    assert _score("computation-006", f"WACC = {_wacc()}") == 1.0


def test_computation_002_uses_absolute_tolerance_and_is_failable():
    # The old `tolerance: 1.0` was read as a 100%-relative band, so every answer
    # (even 0) passed. Absolute tolerance must reject a clearly wrong figure.
    task = TASKS["computation-002"]
    assert task.get("tolerance_abs") is not None
    assert "tolerance" not in task  # the misleading relative band is gone
    wrong = json.dumps({"2021": 1153.0, "2022": 1380.5, "2023": 1615.2})  # the old, wrong values
    assert scorer.score_task(task, {"final_response": wrong})["score"] < 1.0


# ── Scorer unit test: absolute vs relative tolerance ───────────────────────────
def test_absolute_tolerance_semantics():
    exp = {"2021": 753.0}
    assert scorer.score_fuzzy_dict('{"2021": 753.4}', exp, tolerance_abs=1.0)["score"] == 1.0
    assert scorer.score_fuzzy_dict('{"2021": 760.0}', exp, tolerance_abs=1.0)["score"] == 0.0
    # Without an absolute band, tolerance=1.0 is a 100% relative band (the old bug):
    assert scorer.score_fuzzy_dict('{"2021": 0.0}', exp, tolerance=1.0)["score"] == 1.0


# ── debug-002: the functional test must grade predictions, not just actuals ─────
_GOOD_BACKTEST = """
def walk_forward_backtest(revenues, train_window=4):
    results = []
    for i in range(train_window, len(revenues)):
        train = revenues[i - train_window:i]
        actual = revenues[i]
        n = len(train)
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(train) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, train))
        slope = sxy / sxx
        intercept = my - slope * mx
        predicted = slope * train_window + intercept
        results.append((actual, predicted))
    return results
"""

# Returns the right actuals but never fits a model — passed the OLD test.
_LAZY_BACKTEST = """
def walk_forward_backtest(revenues, train_window=4):
    return [(revenues[i], revenues[i]) for i in range(train_window, len(revenues))]
"""


def test_debug_002_accepts_correct_solution():
    code = "```python\n" + _GOOD_BACKTEST + "\n```"
    assert scorer.score_functional(code, TASKS["debug-002"]["test_code"])["score"] == 1.0


def test_debug_002_rejects_prediction_free_solution():
    code = "```python\n" + _LAZY_BACKTEST + "\n```"
    assert scorer.score_functional(code, TASKS["debug-002"]["test_code"])["score"] == 0.0


# ── Fixed "gotcha" code_generation tasks: a prompt-compliant solver must pass ───
# These three tasks previously had a mismatch between what the prompt described
# and what the hidden test demanded (an undocumented 4th arg, an under-specified
# adjustment formula). A reference implementation written strictly from the
# clarified prompt must now score 1.0 — locking the prompt and test together so a
# future prompt edit that re-introduces the gap is caught here.
_SLOAN = """
def compute_sloan_accruals(net_income, cfo, cfi, avg_assets):
    total_accruals = net_income - (cfo + cfi)
    accrual_ratio = total_accruals / avg_assets
    return {"total_accruals": total_accruals,
            "accrual_ratio": accrual_ratio,
            "quality_flag": accrual_ratio > 0.05}
"""

_RECENCY = """
def recency_weighted_forecast(revenues, weights_decay=0.9):
    import numpy as np
    n = len(revenues)
    x = np.arange(n, dtype=float)
    y = np.array(revenues, dtype=float)
    w = np.array([weights_decay ** (n - 1 - i) for i in range(n)], dtype=float)
    W = np.diag(w)
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    return float(beta[0] + beta[1] * n)
"""

_FORECASTER = """
class RevenueForecaster:
    def __init__(self, beat_rate=0.7):
        self.beat_rate = beat_rate
        self._actuals = []
        self._guidance = []
    def add_quarter(self, actual, guidance_low, guidance_high):
        self._actuals.append(actual)
        self._guidance.append((guidance_low, guidance_high))
    def forecast(self):
        import numpy as np
        lo, hi = self._guidance[-1]
        mid = (lo + hi) / 2
        n = len(self._actuals)
        slope, intercept = np.polyfit(np.arange(n), np.array(self._actuals, dtype=float), 1)
        return {
            "naive": self._actuals[-1],
            "guidance_adjusted": mid * (1 + (self.beat_rate - 0.5) * 0.02),
            "trend": float(intercept + slope * n),
        }
"""


def test_codegen_005_sloan_signature_is_satisfiable():
    code = "```python\n" + _SLOAN + "\n```"
    assert scorer.score_functional(code, TASKS["codegen-005"]["test_code"])["score"] == 1.0


def test_codegen_008_recency_weighted_forecast_passes():
    code = "```python\n" + _RECENCY + "\n```"
    assert scorer.score_functional(code, TASKS["codegen-008"]["test_code"])["score"] == 1.0


def test_codegen_009_forecaster_formula_passes():
    code = "```python\n" + _FORECASTER + "\n```"
    assert scorer.score_functional(code, TASKS["codegen-009"]["test_code"])["score"] == 1.0


def test_all_code_generation_tasks_have_room_to_iterate():
    # Phase 3: codegen tasks must allow enough turns to self-test via execute_python.
    for t in TASKS.values():
        if t["category"] == "code_generation":
            assert t["max_turns"] >= 8, f"{t['id']} max_turns={t['max_turns']} (<8)"
