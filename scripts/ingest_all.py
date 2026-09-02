"""Protect the corpus under every scope on the curve. Runs unattended.

    python scripts/ingest_all.py              # all five scopes
    python scripts/ingest_all.py none direct  # named scopes only

Each scope writes to data/protected/<slug>/ and is skipped if already present, so an
interrupted run resumes rather than repaying the API cost of completed scopes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# README step 2 is `cp .env.example .env`; make that instruction true.
from protometer.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

import requests  # noqa: E402

from protometer.ingest import DISCOVERY_URL, ingest  # noqa: E402
from protometer.protect import Protector  # noqa: E402
from protometer.retrieval import build_index  # noqa: E402
from protometer.scopes import CURVE_ORDER, get_scope  # noqa: E402

CORPUS_DIR = ROOT / "data" / "corpus"
PROTECTED_DIR = ROOT / "data" / "protected"
INDEX_DIR = ROOT / "data" / "index"
MANIFEST = PROTECTED_DIR / "ingestion_summary.json"


def _write_manifest(reports: list[dict], status: str, error: str = "") -> None:
    """Persist the run manifest after every scope, not only at the end.

    The end-only write meant a mid-run crash left a traceback as the only record, and worse:
    the committed manifest once described a *single-scope* invocation (the ablation retry)
    while eight per-scope reports sat beside it, the manifest reflected the last partial run,
    not the corpus state. An operator asking "what happened overnight" needs the answer in a
    file with error counts, not in scrollback.
    """
    PROTECTED_DIR.mkdir(parents=True, exist_ok=True)
    # Merge with the manifest already on disk rather than replacing it. A partial invocation
    # (one scope re-ingested after a fix) used to clobber the record of the full run, the
    # committed manifest described a single-scope retry while eight per-scope reports sat
    # beside it. The manifest's job is to describe the corpus *state*, not the last command.
    merged: dict[str, dict] = {}
    try:
        for prior in json.loads(MANIFEST.read_text()).get("reports", []):
            merged[prior.get("scope") or prior.get("scope_name", "?")] = prior
    except (OSError, json.JSONDecodeError):
        pass
    for report in reports:
        merged[report.get("scope") or report.get("scope_name", "?")] = report
    reports = list(merged.values())
    from protometer.persist import atomic_write_json

    atomic_write_json(MANIFEST, {
        "status": status,
        "error": error,
        "scopes_completed": [r.get("scope") or r.get("scope_name", "?") for r in reports],
        # Shape-tolerant: legacy reports carry these as formatted strings, and a sum
        # that silently reads 0 from them misreports the corpus state.
        "total_failures": sum(
            sum(v.values()) if isinstance(v := (r.get("protection_failures") or {}), dict)
            else 0
            for r in reports
        ),
        "total_noops": sum(
            sum(v.values()) if isinstance(v := (r.get("protection_noops") or {}), dict)
            else 0
            for r in reports
        ),
        "reports": reports,
    })

    # Rebuild the compact protection-token manifest the serving guardrail's surrogate-discount
    # ships with (data/protected/ is .dockerignore'd, so the container relies on this manifest, not
    # the bulk artifacts). Best-effort: a manifest failure must never fail an otherwise-good ingest.
    if status == "ok":
        try:
            import importlib.util
            _spec = importlib.util.spec_from_file_location(
                "build_token_manifest", Path(__file__).resolve().parent / "build_token_manifest.py")
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _mod.main()
        except Exception as exc:  # noqa: BLE001
            from protometer.log import get_logger
            get_logger("ingest").warning(
                "token-manifest rebuild skipped: %s: %s", type(exc).__name__, exc)


def _index(scope_slug: str, protected_dir: Path) -> None:
    """Embed the protected narratives for one scope.

    Indexing lives here rather than in a separate script because the two steps are not
    independent: an index is only valid for the corpus it was built from, and every consumer
    (`eval/runner.py`, `measure_semantic_erasure.py`) opens an existing index rather than
    building one. Leaving the build unwired meant `build_index` had no callers at all and a
    missing index was created empty on first search.

    A [[StaleIndexError]] is surfaced with the remedy rather than swallowed: continuing with
    a mismatched index would answer the new corpus's evaluation using the previous corpus's
    chunks.
    """
    from protometer.retrieval import StaleIndexError

    try:
        index = build_index(protected_dir, INDEX_DIR, scope_slug)
    except StaleIndexError as exc:
        raise SystemExit(
            f"[{scope_slug}] {exc}\n"
            f"The corpus changed since this index was built. Remove it and re-run:\n"
            f"  rm -rf data/index/{scope_slug}"
        ) from exc
    print(f"[{scope_slug}] indexed {len(index)} narratives")


def _preflight(scope_names: list[str]) -> None:
    """Fail in seconds on a missing dependency, not minutes into a run.

    Ingestion needs two services, and neither failure is obvious from the error it eventually
    produces. The discovery containers were once silently removed by a Docker Desktop update;
    the next run protected the one scope that needs no discovery, spent minutes on it, and then
    died on `Connection refused` deep inside a stack trace. Credentials fail similarly late.

    `check_determinism.py` already preflights credentials this way, and ingestion, the longest
    and most expensive step, did not.
    """
    missing = [v for v in ("DEV_EDITION_EMAIL", "DEV_EDITION_PASSWORD", "DEV_EDITION_API_KEY")
               if not os.getenv(v)]
    if missing and any(get_scope(n).entities for n in scope_names):
        sys.exit(
            f"Missing credential(s): {', '.join(missing)}\n"
            f"Copy .env.example to .env and fill them in."
        )

    # Only the `none` scope skips discovery, so probe unless that is all that was asked for.
    if not any(get_scope(n).entities for n in scope_names):
        return
    try:
        response = requests.post(
            DISCOVERY_URL,
            params={"score_threshold": 0.6},
            headers={"Content-Type": "text/plain"},
            data=b"Leila Rahman lives at 12 Bridge Road.",
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001, any failure here is the same instruction
        sys.exit(
            f"Data Discovery unreachable at {DISCOVERY_URL}: {exc}\n\n"
            f"Start it with:\n"
            f"  cd vendor-de/data-discovery && docker compose \\\n"
            f"    -f docker-compose.yml -f ../../docker/vendor/discovery.override.yml up -d\n\n"
            f"Override the endpoint with PROTOMETER_DISCOVERY_URL if it runs elsewhere."
        )

    # A provider can fail while the request returns 200, and a dead Context provider means
    # every PERSON passes through unprotected while ingestion reports success.
    for provider in payload.get("providers", []):
        name = provider.get("config_provider", {}).get("name", "?")
        if provider.get("status") != 200:
            sys.exit(
                f"Discovery provider {name!r} is unhealthy (status "
                f"{provider.get('status')}). Restart the discovery stack before ingesting, "
                f"a failed provider silently drops an entire entity class."
            )
    found = set(payload.get("classifications") or {})
    if "PERSON" not in found:
        sys.exit(
            f"Discovery is up but did not detect PERSON in the probe text (found: "
            f"{sorted(found) or 'nothing'}). The Context provider is likely still warming up; "
            f"wait a few seconds and retry."
        )
    print(f"preflight OK, discovery detected {', '.join(sorted(found))}\n")


def main(argv: list[str]) -> int:
    # `--help` is the first thing anyone types. Treating it as a scope name produced
    # `KeyError: "Unknown scope '--help'"`, which reads as a broken script.
    if any(a in ("-h", "--help") for a in argv[1:]):
        print(__doc__)
        return 0

    from protometer.persist import acquire_run_lock

    try:
        _lock = acquire_run_lock(ROOT / "data")  # held for process lifetime  # noqa: F841
    except RuntimeError as exc:
        sys.exit(str(exc))

    scope_names = argv[1:] or list(CURVE_ORDER)
    if not CORPUS_DIR.exists():
        sys.exit("No corpus found. Run: python scripts/build_corpus.py")
    _preflight(scope_names)

    started = time.monotonic()
    reports = []

    for name in scope_names:
        scope = get_scope(name)
        out_dir = PROTECTED_DIR / scope.slug

        if (out_dir / "ingestion_report.json").exists():
            print(f"[{scope.name}] already ingested, skipping protection")
            persisted = json.loads((out_dir / "ingestion_report.json").read_text())
            reports.append(persisted)
            _write_manifest(reports, status="in-progress")
            # Refresh the operational series from the persisted report even on skip, so the
            # Prometheus/Grafana plane is rebuildable from artifacts at $0 (the protect calls
            # are what cost money, not the metric push). Without this, the operational plane
            # could only ever be refreshed by re-paying for a full protect run.
            from protometer.metrics_export import push_from_report

            push_from_report(scope.slug, persisted, domain="aml")
            # Still index: protection and indexing are separate steps, and an interrupted
            # run can leave a fully protected scope with no index at all.
            try:
                _index(scope.slug, out_dir)
            except SystemExit as exc:
                _write_manifest(reports, status="failed",
                                error=f"{scope.name} (index): {str(exc)[:200]}")
                raise
            continue

        print(f"[{scope.name}] {scope.description}")
        # A fresh client per scope: the token cache must not leak across scopes, and the
        # ablation needs its own IV.
        protector = None if not scope.entities else Protector(rotate_iv=scope.break_determinism)

        try:
            report = ingest(CORPUS_DIR, out_dir, scope, protector)
        except (Exception, SystemExit) as exc:  # noqa: BLE001, record, then re-raise
            # The manifest must say the run died and where; a traceback in scrollback is not
            # an operational record.
            _write_manifest(
                reports, status="failed",
                error=f"{scope.name}: {type(exc).__name__}: {str(exc)[:200]}",
            )
            raise
        reports.append(report.to_dict())
        _write_manifest(reports, status="in-progress")
        # The most expensive, most failure-prone stage was the one stage with no
        # experiment-ledger record; its stats lived only in the manifest and scrollback.
        # Operational metrics go to Prometheus (time-series, Grafana-dashboarded): rate,
        # latency, per-scope duration, no-op/failure counts. This is what an operator watches,
        # and it is the wrong shape for MLflow's experiment-comparison model.
        from protometer.metrics_export import push_from_report

        push_from_report(scope.slug, report.to_dict(), domain="aml")
        print(
            f"[{scope.name}] done in {report.seconds:.0f}s, "
            f"entities found={report.entities_found} protected={report.entities_protected}"
        )
        if report.protection_stats:
            stats = report.protection_stats
            print(
                f"[{scope.name}] api_calls={stats['api_calls']} "
                f"values={stats['values_protected']} cache_hits={stats['cache_hits']} "
                f"({stats['cache_hit_rate']:.1%}) retries={stats['retries']} "
                f"api_time={stats['seconds_in_api']:.1f}s"
            )
        if report.entities_skipped:
            print(f"[{scope.name}] SKIPPED (unmapped): {report.entities_skipped}")
        try:
            _index(scope.slug, out_dir)
        except SystemExit as exc:
            # A StaleIndexError exits with instructions; the manifest must not keep saying
            # "in-progress" with no error while the operator reads a traceback.
            _write_manifest(reports, status="failed",
                            error=f"{scope.name} (index): {str(exc)[:200]}")
            raise
        print()

    _write_manifest(reports, status="complete")
    print(f"All scopes complete in {time.monotonic() - started:.0f}s, {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
