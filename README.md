# FinCodeBench

**A coding agent eval suite for financial data tasks**

35 tasks across 6 categories, testing Claude Code's ability to extract structured financial data,
generate correct financial code, run fixed multi-step workflows, work agentically across multiple
tools on open-ended problems, and debug broken models.

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
| extraction      | 5     | exact_json / fuzzy_dict / llm_judge | Parse financial tables, filings, press releases |
| code_generation | 9     | functional (unit tests)| Write correct financial functions |
| computation     | 6     | fuzzy_number / fuzzy_dict | Financial formulas, ratios, WACC, Altman Z |
| workflow        | 5     | llm_judge             | Fixed, pre-specified pipeline (do step 1..4) run through the tool loop |
| agentic         | 7     | llm_judge             | Open-ended: discover data across tools, decide an approach, reconcile messy inputs (incl. building DCF & sum-of-the-parts valuations) |
| debug           | 3     | functional (unit tests)| Find and fix bugs in financial code |

The split between **workflow** and **agentic** is deliberate. A workflow hands the agent the
decomposition — the prompt enumerates the steps; the agent just executes them and self-corrects.
An agentic task hands over only the goal: the data is spread across tools (and some of it is
reachable only via `fetch_filing`, never the local files), it is deliberately messy (missing
periods, mixed units, restatements, duplicates, anomalies), and the agent must decide what to do,
which tool to call next, and when it is done. Skipping the investigation lands on a wrong answer.

---

## Setup

```bash
# 1. Clone / download
git clone https://github.com/yourname/fincodebench
cd fincodebench

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set an API key for your provider of choice
export ANTHROPIC_API_KEY=sk-ant-...      # default provider
# or, for another provider:
# export FINCODEBENCH_PROVIDER=openai
# export OPENAI_API_KEY=sk-...

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
Used for workflow and agentic tasks where output format varies and execution matters.
See `harness/judge.py` for the judge prompt and calibration tooling.

---

## Project structure

```
fincodebench/
├── tasks/
│   └── tasks.json          # 35 task definitions (embedded context, or a tools_data block for agentic tasks)
├── harness/
│   ├── providers.py        # Provider registry + unified Anthropic/OpenAI chat client
│   ├── runner.py           # Executes the model on each task, captures trajectory
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
## Results — claude-opus-4-8 — [date]

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

For **agentic** tasks (scored by `llm_judge`), data lives behind tools instead of in `context`.
Add a `tools_data` block and the runner builds the task's tool set automatically:

```json
{
  "id": "agentic-NNN",
  "category": "agentic",
  "difficulty": "medium",
  "prompt": "An open-ended goal — no numbered steps.",
  "context": null,
  "scoring_type": "llm_judge",
  "rubric": "Score 4 if … Score 0 if no code runs.",
  "max_turns": 8,
  "tools_data": {
    "files":   { "data.csv": "col,...\n..." },
    "filings": { "Acme Inc": { "2024": { "ebitda_usd_m": 260 } } }
  }
}
```

`files` are seeded into the agent's private working directory — reachable via `list_files` /
`read_file` and openable from `execute_python` (which runs in that directory). `filings` are served
only by `fetch_filing(company, year)` and never written to disk, so a task that needs them forces
the model to actually choose that tool. Both keys are optional; omit `tools_data` entirely and a
task gets only `execute_python`, exactly as before.

### Add new scoring methods
Implement a function in `scorer.py` and add a branch to `score_task()`.

### Run against a different model or provider
The runner and judge talk to any supported provider through one interface
(`harness/providers.py`). Anthropic uses its native SDK; every other provider
speaks the OpenAI-compatible chat-completions API.

**Supported providers:** `anthropic` (default), `openai`, `openrouter`,
`deepseek`, `qwen`, `kimi`, `venice`.

From the CLI, select a provider and model via environment variables:

```bash
export FINCODEBENCH_PROVIDER=openai      # default: anthropic
export OPENAI_API_KEY=sk-...             # each provider reads its own key env var
export FINCODEBENCH_MODEL=gpt-4o-mini    # optional — defaults to the provider's default
export FINCODEBENCH_JUDGE_MODEL=gpt-4o   # optional — defaults to the provider's default
python eval.py
```

Per-provider key env vars: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY` (Qwen),
`MOONSHOT_API_KEY` (Kimi), `VENICE_API_KEY`.

From the **web dashboard / API**, runs are bring-your-own-key: choose a provider,
optionally a model, and paste your key. It is sent only with your run request
(header `X-Provider-Api-Key`), billed to your account, and never stored. Adding a
new OpenAI-compatible provider is a single entry in `PROVIDERS` — no other code
changes. Comparing models across providers on the same suite is a useful exercise.

```bash
# Trigger a run on any provider via the API
curl -X POST <url>/runs \
  -H 'Content-Type: application/json' \
  -H 'X-Provider-Api-Key: <your-key>' \
  -d '{"provider":"deepseek","model":"deepseek-chat","categories":["computation"]}'

# Discover supported providers, key hints, and default models
curl <url>/providers
```

---

## Deployment & run history

The Results table needs durable storage to survive restarts and redeploys.
Three tiers, in order of preference:

1. **Postgres (recommended — works on any host, including Render's free tier
   where persistent disks aren't available).** Set `DATABASE_URL` to any
   Postgres connection string; runs are stored in the database, survive
   restarts/spin-downs, and are shared across browsers. Durable free options:
   **Neon** or **Supabase**. (Render's own free Postgres works too but expires
   after a trial window.) No `DATABASE_URL` → this tier is skipped.
2. **Persistent disk (paid plans).** Keep `DATA_DIR=/data` and mount a disk
   there (see `render.yaml`). Runs are stored as JSON files and self-heal into
   the index on boot.
3. **Browser cache (always on).** Completed runs are mirrored to the visitor's
   `localStorage`, so your own runs stay visible even when the server has no
   durable storage — but they aren't shared across devices or users.

With neither (1) nor (2), the server treats run data as ephemeral and logs a
warning on boot; only the browser cache keeps runs visible.

---

## Dependencies

- `anthropic` — Claude API SDK (Anthropic provider)
- `openai` — OpenAI SDK, also used for every OpenAI-compatible provider
  (OpenAI, OpenRouter, DeepSeek, Qwen, Kimi, Venice) via a custom base URL
- `fastapi` + `uvicorn` — the web service
- `psycopg` — optional Postgres run storage (used only when `DATABASE_URL` is set)
- `python3` in PATH — for functional test execution
- Standard library only otherwise (json, subprocess, re, pathlib)

---

## License

MIT
