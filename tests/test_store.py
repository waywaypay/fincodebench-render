"""Storage-backend contract tests.

The same contract must hold for both backends (disk + Postgres) so the API and
dashboard behave identically regardless of which is active. The contract test
runs against whatever backend is configured (disk by default in CI); the
Postgres test runs only when DATABASE_URL points at a reachable database.
"""
import os
import pytest

import api


def _exercise(store):
    rid = "20990101_000000_storetest"
    store.delete_run(rid)  # start clean

    # ── registry round-trip ──────────────────────────────────────────────────
    reg = store.load_registry()
    reg[rid] = {
        "run_id": rid, "status": "queued", "created_at": "2099-01-01T00:00:00",
        "provider": "openai", "model": "m1", "task_ids": ["codegen-001"],
    }
    store.save_registry(reg)
    assert rid in store.load_registry()

    # ── report + results round-trip ──────────────────────────────────────────
    assert store.get_report(rid) is None
    assert store.get_results(rid) is None
    store.put_report(rid, {"provider": "openai", "model": "m1", "overall": {"mean_score": 1.0}})
    store.put_results(rid, [{"task_id": "codegen-001", "model": "m1"}])
    assert store.get_report(rid)["provider"] == "openai"
    assert store.get_results(rid)[0]["task_id"] == "codegen-001"

    # ── a meta-only save must not clobber the report/results ─────────────────
    reg = store.load_registry()
    reg[rid]["status"] = "completed"
    store.save_registry(reg)
    assert store.load_registry()[rid]["status"] == "completed"
    assert store.get_report(rid)["provider"] == "openai"   # still there

    # ── delete removes the run's report/results ──────────────────────────────
    store.delete_run(rid)
    assert store.get_report(rid) is None
    assert store.get_results(rid) is None
    # and drop it from the index
    reg = store.load_registry()
    reg.pop(rid, None)
    store.save_registry(reg)
    assert rid not in store.load_registry()


def test_active_store_contract():
    """Whatever backend the app loaded (disk in CI) satisfies the contract."""
    _exercise(api._STORE)


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")
def test_postgres_backend_contract():
    """Postgres backend specifically — only runs when a database is available."""
    _exercise(api._PgBackend(os.environ["DATABASE_URL"]))
