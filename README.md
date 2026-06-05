# FinCodeBench

**A coding agent eval suite for financial data tasks**

30 tasks across 5 categories, testing Claude Code's ability to extract structured financial data,
generate correct financial code, run multi-step agentic computations, and debug broken models.

---

## Why this exists

Most coding agent benchmarks (SWE-bench, HumanEval) test generic software engineering.
FinCodeBench tests a specific, high-value domain: financial analysis coding — the kind of
work done by quantitative analysts, IR teams, and fintech developers every day.

This is also the domain where model failures are most costly: an off-by-one in a DCF model
or a hallucinated revenue figure isn't an inconvenience, it's a wrong investment decision.

---

## Task categories

| Category        | Count | Scoring method        | What it tests |
|----------------|-------|-----------------------|---------------|
| extraction      | 5     | exact_json / llm_judge | Parse financial tables, filings, press releases |
| code_generation | 9     | functional (unit tests)| Write correct financial functions |
| computation     | 6     | fuzzy_number / fuzzy_dict | Financial formulas, ratios, WACC, Altman Z |
| agentic         | 5     | llm_judge             | Multi-step: fetch + compute + summarize |
| debug           | 3     | functional (unit tests)| Find and fix bugs in financial code |

---

## Setup

```bash
# 1. Clone / download
git clone https://github.com/yourname/fincodebench
cd fincodebench

# 2. Install dependencies
pip install anthropic

# 3. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Create results directory
mkdir -p results/raw
```

---

## Usage

### Run full benchmark
```bash
cd harness
python eval.py
```

### Run specific category only
```bash
python eval.py --category code_generation
python eval.py --category extraction computation
```

### Run a single task (for development)
```bash
python eval.py --task codegen-001
python runner.py --task codegen-001 codegen-002
```

### Score a raw result manually
```bash
python scorer.py results/raw/codegen-001.json
python judge.py results/raw/agentic-001.json
```

### Analyze trajectories and failure modes
```bash
# Summary stats
python trajectory.py --summary

# Interactive failure classification
python trajectory.py --classify
```

### Calibrate your LLM judge
```bash
# After running eval, fill in human_score fields in results/calibration_template.json
# Then:
python judge.py calibrate results/calibration_template.json
```

---

## Scoring methods

**exact_json** — Parse JSON from response, compare to ground truth via equality.
Best for extraction tasks with unambiguous expected outputs.

**fuzzy_number** — Find the closest number in the response, accept if within 2% of expected.
Good for computation tasks where formatting varies.

**fuzzy_dict** — Parse JSON dict, compare each value with tolerance.
Used for multi-field computation outputs (e.g. margin by quarter).

**functional** — Append a unit test suite to the model's generated code, execute it.
Score 1 if tests pass (print PASS + exit 0), 0 otherwise.
The most rigorous method for code generation tasks.

**llm_judge** — Separate Claude call with a structured 0–4 rubric.
Used for agentic tasks where output format varies and execution matters.
See `harness/judge.py` for the judge prompt and calibration tooling.

---

## Project structure

```
fincodebench/
├── tasks/
│   └── tasks.json          # 30 task definitions with embedded context data
├── harness/
│   ├── runner.py           # Executes Claude on each task, captures trajectory
│   ├── scorer.py           # Deterministic scoring (exact, fuzzy, functional)
│   ├── judge.py            # LLM-as-judge for complex/agentic tasks
│   ├── trajectory.py       # Trajectory analysis and failure classification
│   └── eval.py             # Orchestrates full pipeline, generates report
├── results/
│   ├── raw/                # Per-task JSON: trajectory + score
│   ├── batch_*.json        # Full run results
│   ├── report_*.json       # Summary reports
│   └── calibration_*.json  # Human vs judge calibration data
└── README.md
```

---

## Key findings (fill in after running)

After running the benchmark, document your findings here. This is the most important
part for demonstrating PM eval skills — the ability to synthesize model behavior
into actionable observations.

Template:

```
## Results — claude-opus-4-5 — [date]

Overall pass rate: XX%
Strongest category: [category] (XX%)
Weakest category: [category] (XX%)

Top 3 failure modes:
1. [mode]: [count] tasks — [one-sentence explanation]
2. [mode]: [count] tasks
3. [mode]: [count] tasks

Behavior I would change in the model:
- [observation 1]
- [observation 2]

If I were Anthropic's PM for Claude Code, I would prioritize:
- [priority 1]
- [priority 2]
```

---

## Extending the benchmark

### Add new tasks
Edit `tasks/tasks.json`. Follow the schema:

```json
{
  "id": "category-NNN",
  "category": "code_generation",
  "difficulty": "easy|medium|hard",
  "prompt": "...",
  "context": null,
  "scoring_type": "functional",
  "test_code": "assert ...\nprint('PASS')",
  "max_turns": 5
}
```

### Add new scoring methods
Implement a function in `scorer.py` and add a branch to `score_task()`.

### Run against a different model
Change `MODEL` in `runner.py`. Re-run. Compare reports.
Comparing Claude models (opus vs sonnet vs haiku) on the same suite is a useful exercise.

---

## Dependencies

- `anthropic` — Claude API SDK
- `python3` in PATH — for functional test execution
- Standard library only otherwise (json, subprocess, re, pathlib)

---

## License

MIT
