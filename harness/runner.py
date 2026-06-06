"""
FinCodeBench Runner
Executes Claude against financial coding tasks and captures full trajectories.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import providers

from pricing import compute_cost

# ── Config ──────────────────────────────────────────────────────────────────
# Provider + model are configurable via env (FINCODEBENCH_PROVIDER /
# FINCODEBENCH_MODEL). MODEL defaults to the selected provider's default model.
PROVIDER, _PROVIDER_CFG = providers.resolve_provider(os.environ.get("FINCODEBENCH_PROVIDER"))
MODEL = os.environ.get("FINCODEBENCH_MODEL") or _PROVIDER_CFG["default_model"]
MAX_TOKENS = 4096
RESULTS_DIR = Path("results")
TASKS_FILE = Path("tasks/tasks.json")

# Built from the provider's key env var for direct CLI use. The web service
# overrides this per run with the caller's own key (bring-your-own-key), so a
# missing env key must not break import — providers falls back to a placeholder.
client = providers.client_from_env(PROVIDER)


# ── Tool definitions ─────────────────────────────────────────────────────────
# `execute_python` is offered to every task. Tasks that declare a `tools_data`
# block additionally get data-access tools — `list_files` / `read_file` over a
# private working directory seeded with the task's files, and `fetch_filing`
# over a canned filings service. Exposing these per task (rather than globally)
# is what makes the agentic tasks require genuine action-selection: the model
# has to decide which tool to call, in what order, based on what it discovers —
# and some data is reachable only through `fetch_filing`, never the filesystem.
EXECUTE_PYTHON_TOOL = {
    "name": "execute_python",
    "description": (
        "Execute Python code and return stdout/stderr. Use this to run "
        "calculations, parse data, or test your code. Any data files provided "
        "for this task are in the current working directory, so you can open "
        "them by name, e.g. open('data.csv')."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"}
        },
        "required": ["code"],
    },
}

LIST_FILES_TOOL = {
    "name": "list_files",
    "description": "List the data files available in your working directory for this task.",
    "input_schema": {"type": "object", "properties": {}},
}

READ_FILE_TOOL = {
    "name": "read_file",
    "description": "Read and return the contents of one data file in your working directory, by name.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "File name to read"}},
        "required": ["path"],
    },
}

FETCH_FILING_TOOL = {
    "name": "fetch_filing",
    "description": (
        "Look up a company's reported financial data for a fiscal year from the "
        "filings service. Returns a JSON record, or an error if no filing exists "
        "for that company and year. Some data is available only here, not in the "
        "local files."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company": {"type": "string", "description": "Company name"},
            "year": {"type": "string", "description": "Fiscal year, e.g. '2024'"},
        },
        "required": ["company", "year"],
    },
}

# Real-data tools for agentic_real tasks. These call live network services but
# serve pre-captured snapshot responses first, giving reproducibility without
# sacrificing authenticity. `source` in the trajectory trace shows whether a
# snapshot or a live call answered each request.
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search for earnings call transcripts, SEC filings (10-K, 10-Q, 8-K), "
        "investor day and conference presentations, analyst reports, and other "
        "public financial information. Returns up to 8 relevant documents with "
        "title, source, and a content excerpt. Use this for information not "
        "available in local files or the filing service — such as historical "
        "precedents, management commentary from calls, or industry comparables."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (e.g. company name, topic, document type)",
            }
        },
        "required": ["query"],
    },
}

FETCH_SEC_FILING_TOOL = {
    "name": "fetch_sec_filing",
    "description": (
        "Retrieve key excerpts from a company's SEC filing (10-K, 10-Q, etc.). "
        "Provide the company name, form type, and fiscal year. Returns a filing excerpt "
        "containing financial tables and key disclosures relevant to your query."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company": {"type": "string", "description": "Company name, e.g. 'Microsoft' or 'Apple'"},
            "form_type": {"type": "string", "description": "SEC form type, e.g. '10-K' or '10-Q'"},
            "year": {"type": "string", "description": "Fiscal year, e.g. '2024'"},
        },
        "required": ["company", "form_type", "year"],
    },
}
# Default tool set for tasks that declare no data tools (kept as a module global
# for back-compat with anything importing runner.TOOLS).
TOOLS = [EXECUTE_PYTHON_TOOL]

MAX_READ_FILE_CHARS = 6000


# ── Tool executors ─────────────────────────────────────────────────────────────
def execute_python(code: str, timeout: int = 30, cwd: Optional[str] = None) -> str:
    """Run Python code in a subprocess, return combined stdout+stderr. When `cwd`
    is given the code runs there, so relative file opens resolve to the task's
    private working directory."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmpfile = f.name

    try:
        result = subprocess.run(
            ['python3', tmpfile],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if err:
            out = out + f"\n[STDERR]\n{err}" if out else f"[STDERR]\n{err}"
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[ERROR] Execution timed out after {timeout}s"
    except Exception as e:
        return f"[ERROR] {str(e)}"
    finally:
        try:
            os.unlink(tmpfile)
        except Exception:
            pass


def _list_files(workdir: str) -> str:
    """Render a directory listing (name + byte size) for the read_file tool."""
    try:
        names = sorted(os.listdir(workdir))
    except Exception as e:
        return f"[ERROR] {e}"
    if not names:
        return "(no files)"
    lines = []
    for n in names:
        try:
            lines.append(f"{n} ({os.path.getsize(os.path.join(workdir, n))} bytes)")
        except Exception:
            lines.append(n)
    return "Files available in the working directory:\n" + "\n".join(lines)


def _read_file(workdir: str, path: str) -> str:
    """Read one file from the task's working directory. Names are flattened to a
    basename so a task can't be coaxed into reading outside its sandbox."""
    if not path:
        return "[ERROR] read_file requires a 'path'."
    safe = os.path.basename(str(path))
    full = os.path.join(workdir, safe)
    if not os.path.isfile(full):
        return f"[ERROR] No such file: {safe}"
    try:
        with open(full) as f:
            data = f.read()
    except Exception as e:
        return f"[ERROR] {e}"
    if len(data) > MAX_READ_FILE_CHARS:
        data = data[:MAX_READ_FILE_CHARS] + "\n…[file truncated]…"
    return data


def _fetch_filing(filings: dict, company: str, year) -> str:
    """Serve a canned filing record. Data here is never written to disk, so a
    task that needs it forces the model to actually choose this tool."""
    company = (company or "").strip()
    year = str(year).strip()
    rec = (filings.get(company) or {}).get(year)
    if rec is None:
        return (f"[ERROR] No filing found for company={company!r}, year={year!r}. "
                "Check the exact company name and fiscal year.")
    return json.dumps(rec, indent=2)


def _web_search(web_results: list, query: str) -> str:
    """Return canned web search results ranked by keyword overlap with the query.

    Each document in web_results is a dict with 'title', 'source', and 'content',
    plus an optional 'keywords' field of concept synonyms (ticker symbols,
    'days sales outstanding' for DSO, 'channel loading', etc.). The keywords are
    folded into the match text but never rendered, so that scoring depends on the
    model's analysis rather than on guessing the exact wording in a document — a
    reasonable concept query surfaces the relevant documents. The model never sees
    this logic."""
    if not query:
        return "(no query provided)"
    if not web_results:
        return "(no web results available for this task)"

    query_terms = set(query.lower().split())
    scored: list[tuple[int, dict]] = []
    for doc in web_results:
        searchable = (
            doc.get("title", "") + " " +
            doc.get("source", "") + " " +
            doc.get("keywords", "") + " " +
            doc.get("content", "")
        ).lower()
        hits = sum(1 for t in query_terms if t in searchable)
        if hits > 0:
            scored.append((hits, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:8]

    if not top:
        return "(no results found — try a broader or different query)"

    parts: list[str] = []
    for rank, (_, doc) in enumerate(top, 1):
        title = doc.get("title", "(no title)")
        source = doc.get("source", "")
        content = doc.get("content", "").strip()
        if len(content) > 1500:
            content = content[:1500] + "\n…[excerpt truncated]…"
        header = f"Result {rank}: {title}"
        if source:
            header += f"\nSource: {source}"
        parts.append(f"{header}\n\n{content}")

    return "\n\n---\n\n".join(parts)


def _web_search_live(query: str, snapshots: dict, timeout: int = 10) -> tuple:
    """Search the web for agentic_real tasks. Returns (result_str, source).
    Serves from snapshot cache first; falls back to DuckDuckGo for cache misses."""
    query = (query or "").strip()
    if query in snapshots:
        return snapshots[query], "snapshot"
    try:
        import urllib.parse
        import urllib.request
        params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": "1"})
        req = urllib.request.Request(
            f"https://api.duckduckgo.com/?{params}",
            headers={"User-Agent": "FinCodeBench/1.0 (benchmark research)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        parts = []
        if data.get("AbstractText"):
            parts.append(f"Summary: {data['AbstractText']}")
            if data.get("AbstractSource"):
                parts.append(f"Source: {data['AbstractSource']}")
        for topic in (data.get("RelatedTopics") or [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(f"- {topic['Text']}")
        return ("\n".join(parts) if parts else "(no results — try a more specific query)"), "live"
    except Exception as e:
        return f"[ERROR] Web search failed: {e}", "error"


def _fetch_sec_filing(company: str, form_type: str, year: str, snapshots: dict, timeout: int = 15) -> tuple:
    """Fetch a SEC filing excerpt. Returns (result_str, source) where source is 'snapshot' or 'live'.
    Snapshot key format: '{company}:{form_type}:{year}', e.g. 'Microsoft:10-K:2024'.
    Live fallback queries EDGAR EFTS and returns filing metadata only (not full content)."""
    company = (company or "").strip()
    form_type = (form_type or "10-K").strip()
    year = str(year or "").strip()
    cache_key = f"{company}:{form_type}:{year}"
    if cache_key in snapshots:
        return snapshots[cache_key], "snapshot"
    try:
        import urllib.parse
        import urllib.request
        params = urllib.parse.urlencode({
            "q": f'"{company}"',
            "forms": form_type,
            "dateRange": "custom",
            "startdt": f"{year}-01-01",
            "enddt": f"{int(year) + 1}-06-30",
        })
        req = urllib.request.Request(
            f"https://efts.sec.gov/LATEST/search-index?{params}",
            headers={"User-Agent": "FinCodeBench/1.0 (benchmark; contact: research@fincodebench.io)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        hits = (data.get("hits") or {}).get("hits") or []
        if not hits:
            return (
                f"[ERROR] No {form_type} filing found for {company!r} in {year}. "
                "Verify the exact company name or try the ticker symbol."
            ), "live"
        hit = hits[0].get("_source") or {}
        meta = {
            "company": hit.get("entity_name", company),
            "form_type": hit.get("form_type", form_type),
            "filed": hit.get("file_date"),
            "period": hit.get("period_of_report"),
            "accession_no": hit.get("accession_no"),
            "note": (
                "Live query — metadata only. Detailed financial tables are not included. "
                "Use execute_python with the accession number to fetch specific sections, "
                "or check that the snapshot key exactly matches the company name."
            ),
        }
        return json.dumps(meta, indent=2), "live"
    except Exception as e:
        return f"[ERROR] SEC EDGAR query failed: {e}", "error"


def build_task_tools(task: dict):
    """Build the (tools, executors, cleanup) triple for one task.

    Tasks with no `tools_data` get just execute_python running in the default
    working directory — identical to the original behavior. Tasks that declare
    `tools_data` get a private temp working directory seeded with their `files`,
    plus the data-access tools, so the model must discover and choose among them.
    `cleanup` removes the temp directory and must be called when the task ends.
    """
    data = task.get("tools_data") or {}
    files = data.get("files") or {}
    filings = data.get("filings") or {}
    web_results = data.get("web_results") or []

    workdir = None
    if files:
        workdir = tempfile.mkdtemp(prefix="fcb_task_")
        for name, content in files.items():
            safe = os.path.basename(str(name))
            text = content if isinstance(content, str) else json.dumps(content, indent=2)
            with open(os.path.join(workdir, safe), "w") as f:
                f.write(text)

    tools = [EXECUTE_PYTHON_TOOL]
    executors = {"execute_python": lambda inp: execute_python(inp.get("code", ""), cwd=workdir)}

    if files:
        tools += [LIST_FILES_TOOL, READ_FILE_TOOL]
        executors["list_files"] = lambda inp: _list_files(workdir)
        executors["read_file"] = lambda inp: _read_file(workdir, inp.get("path", ""))
    if filings:
        tools.append(FETCH_FILING_TOOL)
        executors["fetch_filing"] = lambda inp: _fetch_filing(filings, inp.get("company", ""), inp.get("year", ""))
    if web_results:
        tools.append(WEB_SEARCH_TOOL)
        executors["web_search"] = lambda inp: _web_search(web_results, inp.get("query", ""))

    # Real-data tools for agentic_real tasks. Responses are served from the
    # task's snapshot dict (for reproducibility) with live network fallback.
    # The default-argument capture (_ws, _ss) prevents late-binding closure bugs.
    if "snapshots" in data:
        snaps = data["snapshots"]
        web_snaps = snaps.get("web_search") or {}
        sec_snaps = snaps.get("fetch_sec_filing") or {}
        tools.append(WEB_SEARCH_TOOL)
        tools.append(FETCH_SEC_FILING_TOOL)
        executors["web_search"] = (
            lambda inp, _ws=web_snaps: _web_search_live(inp.get("query", ""), _ws)
        )
        executors["fetch_sec_filing"] = (
            lambda inp, _ss=sec_snaps: _fetch_sec_filing(
                inp.get("company", ""), inp.get("form_type", "10-K"), inp.get("year", ""), _ss
            )
        )

    def cleanup():
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    return tools, executors, cleanup


# Back-compat dispatch table for the default tool set (execute_python only).
TOOL_EXECUTORS = {
    "execute_python": lambda inp: execute_python(inp.get("code", "")),
}


# ── Task runner ───────────────────────────────────────────────────────────────
def run_task(task: dict, verbose: bool = True) -> dict:
    """
    Run a single task against Claude.
    Returns a result dict with trajectory, final_response, turns, and metadata.
    """
    task_id = task["id"]
    if verbose:
        print(f"\n{'─'*60}")
        print(f"Running: {task_id} ({task['category']} / {task['difficulty']})")

    # Build initial prompt
    prompt = task["prompt"]
    if task.get("context"):
        prompt = f"**Context / Data:**\n```\n{task['context']}\n```\n\n**Task:**\n{prompt}"

    # Neutral message history (provider-agnostic — see providers.ChatClient)
    messages = [{"role": "user", "content": prompt}]
    trajectory = []
    final_response = ""
    error = None
    max_turns = task.get("max_turns", 8)

    # Per-task tool set: every task gets execute_python; tasks with a
    # `tools_data` block also get list_files / read_file / fetch_filing over a
    # private working directory (removed by cleanup() once the task finishes).
    tools, executors, cleanup = build_task_tools(task)

    start_time = time.time()
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    for turn in range(1, max_turns + 1):
        try:
            response = client.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                tools=tools,
                messages=messages
            )
        except Exception as e:
            error = str(e)
            if verbose:
                print(f"  [API ERROR turn {turn}] {e}")
            break

        # Accumulate token usage across turns for cost reporting
        u = getattr(response, "usage", None)
        if u is not None:
            for k in usage:
                usage[k] += getattr(u, k, 0) or 0

        # Record this turn (one text block, then any tool_use blocks)
        traj_entry = {
            "turn": turn,
            "stop_reason": response.stop_reason,
            "blocks": []
        }
        if response.text:
            traj_entry["blocks"].append({"type": "text", "text": response.text})
            if verbose:
                preview = response.text[:120].replace('\n', ' ')
                print(f"  [turn {turn} text] {preview}...")
        for tc in response.tool_calls:
            traj_entry["blocks"].append({
                "type": "tool_use",
                "name": tc["name"],
                "input": tc["input"],
                "id": tc["id"]
            })
            if verbose:
                preview = str(tc["input"])[:80]
                print(f"  [turn {turn} tool_use] {tc['name']}: {preview}...")

        trajectory.append(traj_entry)

        # Append assistant turn to neutral history
        messages.append({
            "role": "assistant",
            "text": response.text,
            "tool_calls": response.tool_calls
        })

        # Stop if no tool calls
        if response.stop_reason == "end_turn":
            final_response = response.text
            if verbose:
                print(f"  [done in {turn} turns]")
            break

        # Execute tool calls and append results
        if response.stop_reason == "tool_use" and response.tool_calls:
            tool_results = []
            for tc in response.tool_calls:
                executor = executors.get(tc["name"])
                if executor:
                    t_call = time.time()
                    raw = executor(tc["input"])
                    call_ms = round((time.time() - t_call) * 1000)
                    # Real tools return (output, trace_dict); synthetic tools return a str.
                    if isinstance(raw, tuple) and len(raw) == 2:
                        tool_output, trace_extra = raw[0], raw[1]
                    else:
                        tool_output, trace_extra = raw, {}
                    trace = {"timestamp_utc": datetime.utcnow().isoformat() + "Z",
                             "latency_ms": call_ms}
                    if isinstance(trace_extra, dict):
                        trace.update(trace_extra)
                else:
                    tool_output = f"[ERROR] Unknown tool: {tc['name']}"
                    trace = {"timestamp_utc": datetime.utcnow().isoformat() + "Z"}

                # Record in trajectory — trace carries timing + source for every call
                traj_entry["blocks"].append({
                    "type": "tool_result",
                    "name": tc["name"],
                    "result": tool_output,
                    "trace": trace,
                })

                if verbose:
                    preview = tool_output[:100].replace('\n', ' ')
                    print(f"  [turn {turn} tool_result] {preview}")

                tool_results.append({
                    "id": tc["id"],
                    "name": tc["name"],
                    "output": tool_output
                })

            messages.append({"role": "tool", "results": tool_results})

    # Tear down the task's private working directory, if any.
    cleanup()

    elapsed = time.time() - start_time
    cost_usd = compute_cost(
        MODEL,
        usage["input_tokens"], usage["output_tokens"],
        usage["cache_creation_input_tokens"], usage["cache_read_input_tokens"],
    )

    return {
        "task_id": task_id,
        "model": MODEL,
        "category": task["category"],
        "difficulty": task["difficulty"],
        "scoring_type": task["scoring_type"],
        "final_response": final_response,
        "trajectory": trajectory,
        "turns": len(trajectory),
        "elapsed_seconds": round(elapsed, 2),
        "usage": usage,
        "cost_usd": cost_usd,
        "error": error,
        "provider": PROVIDER,
        "timestamp": datetime.utcnow().isoformat()
    }


# ── Batch runner ──────────────────────────────────────────────────────────────
def run_benchmark(
    task_ids: Optional[list] = None,
    categories: Optional[list] = None,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None
) -> list:
    """
    Run all (or filtered) tasks, save raw results to results/raw/.
    Returns list of raw result dicts.
    """
    with open(TASKS_FILE) as f:
        tasks = json.load(f)

    # Filter
    if task_ids:
        tasks = [t for t in tasks if t["id"] in task_ids]
    if categories:
        tasks = [t for t in tasks if t["category"] in categories]

    print(f"\nRunning {len(tasks)} tasks on {PROVIDER} / {MODEL}")

    raw_dir = RESULTS_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(tasks)
    for i, task in enumerate(tasks):
        if progress_callback:
            try:
                progress_callback(i, total, task["id"])
            except Exception:
                pass
        result = run_task(task, verbose=verbose)
        results.append(result)

        # Save immediately — don't lose progress
        out_path = raw_dir / f"{task['id']}.json"
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)

    if progress_callback:
        try:
            progress_callback(total, total, None)
        except Exception:
            pass

    # Save full batch
    batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    batch_path = RESULTS_DIR / f"batch_{batch_id}.json"
    with open(batch_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results → {batch_path}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FinCodeBench runner")
    parser.add_argument("--task", nargs="+", help="Run specific task IDs")
    parser.add_argument("--category", nargs="+",
                        choices=["extraction", "code_generation", "computation", "workflow", "agentic", "agentic_real", "debug"],
                        help="Run only specific categories")
    parser.add_argument("--quiet", action="store_true", help="Suppress turn-by-turn output")
    args = parser.parse_args()

    results = run_benchmark(
        task_ids=args.task,
        categories=args.category,
        verbose=not args.quiet
    )
    print(f"\n{'='*60}")
    print(f"Completed {len(results)} tasks.")
    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"Errors: {len(errors)} — {[e['task_id'] for e in errors]}")
