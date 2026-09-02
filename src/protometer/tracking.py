"""Experiment tracking, every run recorded, comparable, and reproducible.

This project has now had three measurements invalidated after the fact: a shared cache that
made one scope's numbers another's, a corpus artifact that inflated classifier
accuracy, and a detector confound that turned into a fake utility cliff.
Each was caught by chance rather than by design.

The common thread is that results were compared **across time** without recording what else had
changed. A results file says what the score was; it does not say which corpus, which code, which
detection ledger, or which protection scope produced it. Tracking closes that gap: every run
carries the parameters needed to know whether two numbers are comparable at all.

Runs go to the local MLflow **server** (Docker, `ghcr.io/mlflow/mlflow`, UI at
http://localhost:5001) when it is reachable, and fall back to the same SQLite file the server
itself is backed by when it is not. Both paths write to one store,
`docker/observability/mlflow/store/mlflow.db` (see DEFAULT_TRACKING_DIR / PROTOMETER_MLFLOW_STORE_DIR) -
so runs recorded while the server was down appear in the UI the moment it returns. Tracking is
deliberately **optional**: if MLflow is unavailable the pipeline runs unchanged, because a
measurement harness that cannot run without its telemetry is worse than one without telemetry.

Start the server with `cd docker/observability/mlflow && docker compose up -d`
(compose file: docker/observability/mlflow/compose.yml).
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator

from protometer.log import get_logger

_log = get_logger("tracking")

# SQLite-backed store in the repo. MLflow 3.x put the plain-file backend into maintenance mode
# and refuses it by default; SQLite is also the better choice here, since runs become queryable
# with ordinary SQL while remaining a single portable file a reviewer can open.
# The MLflow store lives under docker/observability/mlflow/store (with the other docker stacks), not under data/:
# it is an observability backing store, not corpus/eval data. Overridable via
# PROTOMETER_MLFLOW_STORE_DIR for a relocated or remote layout.
DEFAULT_TRACKING_DIR = Path(
    os.getenv("PROTOMETER_MLFLOW_STORE_DIR")
    or (Path(__file__).resolve().parents[2] / "docker" / "observability" / "mlflow" / "store")
)
TRACKING_DB = "mlflow.db"

# The server URI, overridable for a remote deployment. Host port 5001 because macOS AirPlay
# listens on 5000 and answers HTTP, which would make a dead server look alive.
from protometer import settings as _settings

DEFAULT_SERVER_URI = _settings.mlflow_uri()


def _server_reachable(uri: str, timeout: float = 2.0) -> bool:
    """One cheap probe, so a down server costs two seconds once instead of a hang per call."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{uri}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001, any failure means the same thing: use the fallback
        return False


def _git_revision() -> str:
    """Current commit, so a run can be tied to the code that produced it."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "uncommitted"
    except Exception:  # noqa: BLE001, tracking must never break the run
        return "unknown"


def _corpus_fingerprint(corpus_dir: Path) -> str:
    """Hash of the corpus, so runs over different data are never silently compared."""
    import hashlib

    try:
        return hashlib.sha256(
            (corpus_dir / "transactions.json").read_bytes()
            + (corpus_dir / "narratives.json").read_bytes()
        ).hexdigest()[:12]
    except OSError:
        return "unknown"


def corpus_source_fingerprint(corpus_dir: Path) -> str:
    """Full-corpus fingerprint (all five files), the same key the eval runner and ingest use.

    Distinct from `_corpus_fingerprint` above (which hashes only the two prompt-input files):
    this is the cross-stage join key that ties a training run and a model version to the
    exact corpus state they were built from.
    """
    import hashlib

    digest = hashlib.sha256()
    try:
        for name in ("transactions.json", "narratives.json", "alerts.json",
                     "ground_truth.json", "parties.json"):
            digest.update((corpus_dir / name).read_bytes())
    except OSError:
        return "unknown"
    return digest.hexdigest()[:12]


def _latest_model_version(tracker: "Tracker", model_name: str) -> str | None:
    """The highest version number of a registered model, or None if unavailable."""
    if not tracker.enabled or tracker._mlflow is None:
        return None
    try:
        client = tracker._mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None
        return max(versions, key=lambda v: int(v.version)).version
    except Exception:  # noqa: BLE001
        return None


class Tracker:
    """Thin wrapper over MLflow that degrades to a no-op when it is unavailable.

    The no-op path matters: an evaluation that fails because a tracking server is down has
    turned observability into a dependency, which is the opposite of what it is for.
    """

    def __init__(
        self,
        experiment: str,
        tracking_dir: Path = DEFAULT_TRACKING_DIR,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and os.getenv("PROTOMETER_NO_TRACKING") != "1"
        self._mlflow = None

        if not self.enabled:
            return
        try:
            import mlflow

            tracking_dir.mkdir(parents=True, exist_ok=True)
            # Server first, shared-SQLite fallback second. Same store either way, so no run
            # is lost to the server being down, it just isn't visible in the UI until the
            # server returns.
            if _server_reachable(DEFAULT_SERVER_URI):
                mlflow.set_tracking_uri(DEFAULT_SERVER_URI)
                self.server_uri = DEFAULT_SERVER_URI
            else:
                mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / TRACKING_DB}")
                self.server_uri = None
                _log.warning(
                    "MLflow server not reachable at %s; recording to SQLite "
                    "(runs appear in the UI when the server is up)", DEFAULT_SERVER_URI
                )
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow
        except Exception as exc:  # noqa: BLE001, never let tracking abort a run
            # Reported rather than swallowed: silent disablement means a run appears tracked
            # and is not, which is exactly the failure this module exists to prevent.
            _log.warning("disabled: %s: %s", type(exc).__name__, str(exc)[:120])
            self.enabled = False

    @contextlib.contextmanager
    def run(self, name: str, params: dict[str, Any] | None = None) -> Iterator["Tracker"]:
        """Open a tracked run, stamping provenance every time.

        Git revision and corpus fingerprint are recorded automatically because those are
        precisely the fields whose absence caused earlier results to be compared when they
        should not have been.
        """
        if not self.enabled or self._mlflow is None:
            yield self
            return

        # The setup pair was the one unguarded path left: a locked SQLite file at run-open
        # raised into the pipeline while every later call degraded politely, the exact
        # inversion the module docstring promises to avoid. If the run cannot open, tracking
        # disables for this run and the pipeline proceeds untracked but alive.
        try:
            run_context = self._mlflow.start_run(run_name=name)
        except Exception as exc:  # noqa: BLE001, tracking must never abort a run
            _log.warning("start_run failed: %s: %s", type(exc).__name__, str(exc)[:100])
            yield self
            return
        with run_context:
            # The git commit goes in MLflow's reserved tag, where the UI and search expect
            # it (`mlflow.source.git.commit` is what run-comparison views read), rather
            # than in a homemade param.
            from protometer.persist import RUN_ID

            self._safe(
                self._mlflow.set_tags,
                {
                    "mlflow.source.git.commit": _git_revision(),
                    # The join key across MLflow, Langfuse, and artifact JSON: one id
                    # per process, so "which prompts produced this metric" is a query,
                    # not wall-clock archaeology.
                    "protometer.run_id": RUN_ID,
                },
            )
            # The cross-plane join key MUST be the 5-file fingerprint every other plane stamps
            # (eval artifacts, champion tags, ingest/Prometheus source_fingerprint). Stamping the
            # 2-file `_corpus_fingerprint` here gave MLflow a `corpus_fingerprint` that never
            # matched the ingest/artifact value, so the operational<->experiment join silently
            # failed (a reviewer sees two different hashes for the same corpus). Keep the 2-file
            # value too, but under a DISTINCT name so the join key means one thing everywhere.
            _corpus_dir = Path(__file__).resolve().parents[2] / "data" / "corpus"
            self._safe(
                self._mlflow.log_params,
                {
                    "corpus_fingerprint": corpus_source_fingerprint(_corpus_dir),
                    "prompt_fingerprint": _corpus_fingerprint(_corpus_dir),
                    **{k: str(v)[:250] for k, v in (params or {}).items()},
                },
            )
            yield self

    def _safe(self, operation, *args, **kwargs) -> None:
        """Run one MLflow call, degrading loudly rather than aborting the pipeline.

        Init was guarded but every mid-run call was not, so a locked SQLite file or full disk
        halfway through an evaluation raised into the run, turning telemetry into a
        dependency, the exact inversion this module's docstring promises to avoid.
        """
        try:
            operation(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001, tracking must never abort a run
            _log.warning("%s failed: %s: %s", operation.__name__, type(exc).__name__, str(exc)[:100])

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled or self._mlflow is None:
            return
        import math

        numeric: dict[str, float] = {}
        dropped: list[str] = []
        for k, v in metrics.items():
            # Reject bools (a bool is an int), non-numbers, and NaN/inf, which MLflow rejects
            # anyway, but silently at the call boundary. Record what was dropped so an absent
            # metric is a visible decision, not an invisible omission a reviewer must guess at.
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                numeric[k] = float(v)
            else:
                dropped.append(k)
        if dropped:
            _log.debug("log_metrics skipped non-finite/non-scalar keys: %s", dropped)
        if numeric:
            self._safe(self._mlflow.log_metrics, numeric, step=step)

    def log_nested(self, prefix: str, values: dict[str, Any]) -> None:
        """Flatten a nested metric dict, e.g. precision_at_k, into scalar metrics."""
        if not self.enabled or self._mlflow is None:
            return
        flat: dict[str, float] = {}
        for key, value in values.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    if isinstance(inner_value, (int, float)):
                        flat[f"{prefix}.{key}.{inner_key}"] = float(inner_value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                flat[f"{prefix}.{key}"] = float(value)
        if flat:
            self._safe(self._mlflow.log_metrics, flat)

    def log_model(self, model: Any, name: str, input_example: Any) -> str | None:
        """Log the fitted model itself, with signature, MLflow's own provenance unit.

        The project previously hand-rolled provenance as `classifier_hash` while never
        calling `log_model`: reproducing a ranking meant refitting from scratch and hoping
        the hash matched. A logged model with an inferred signature gives a versioned
        `models:/` URI the registry can serve back byte-for-byte, which is what the
        five-year reconstruction obligation actually asks for. `classifier_hash` stays as a
        cross-check tag on the run; the two mechanisms verify each other.

        Returns the exact registered version this call created (or None if logging was off or
        failed), so the caller governs *that* version rather than re-querying "latest" in a
        second, unordered call, a query that can bless a different version under a concurrent
        registration.
        """
        if not self.enabled or self._mlflow is None:
            return None
        self._warn_if_artifacts_unreachable()
        try:
            import numpy as np
            from mlflow.models import infer_signature

            example = np.asarray(input_example)
            # Signature must describe what pyfunc actually SERVES. The sklearn flavor's
            # default pyfunc calls predict() -> (n,) class labels, so inferring against
            # predict_proba() -> (n, 2) mis-declared the output. Infer against predict().
            signature = infer_signature(example, model.predict(example))
            info = self._mlflow.sklearn.log_model(
                model,
                name=name,
                signature=signature,
                input_example=example,
                registered_model_name=f"protometer-{name}",
            )
            # The ModelInfo carries the version this exact call registered; prefer it over a
            # follow-up "latest" lookup that a concurrent run could have advanced.
            version = getattr(info, "registered_model_version", None)
            return str(version) if version is not None else None
        except Exception as exc:  # noqa: BLE001, tracking must never abort a run
            _log.warning("log_model failed: %s: %s", type(exc).__name__, str(exc)[:100])
            return None

    def govern_model(self, name: str, version: str, tags: dict[str, str],
                     alias: str = "champion", champion_if_best: bool = False) -> None:
        """Apply MLflow-3 governance to a model version: descriptive tags + a champion alias.

        MLflow 3 deprecated the None->Staging->Production->Archived *stages* in favour of
        aliases and tags, which is what this uses. Each version is tagged with the facts that
        identify it (classifier hash, corpus fingerprint, AP), and the best-AP build per scope
        gets an alias (`champion` by default) that `models:/protometer-<scope>@champion`
        resolves to, so a consumer always fetches the blessed version without hardcoding a
        number. Superseded versions are aliased `archived` by the promote script.

        `champion_if_best`: when True, the version is ALWAYS tagged, but the `alias` moves to it
        only if its `average_precision` tag is >= the incumbent champion's. This makes "champion"
        mean best-AP, not newest: a retrain that regressed (worse corpus, worse features) does not
        silently demote a better model. The reconcile script (`govern_models.py`) applies the same
        best-AP policy across all versions.
        """
        if not self.enabled or self._mlflow is None:
            return
        try:
            client = self._mlflow.MlflowClient()
            for k, v in tags.items():
                client.set_model_version_tag(name, version, k, str(v)[:250])
            if champion_if_best and not self._beats_incumbent(client, name, alias, tags):
                _log.info("govern_model: v%s did not beat @%s champion; alias unchanged",
                          version, alias)
                return
            client.set_registered_model_alias(name, alias, version)
        except Exception as exc:  # noqa: BLE001
            _log.warning("govern_model failed: %s: %s", type(exc).__name__, str(exc)[:100])

    @staticmethod
    def _beats_incumbent(client, name: str, alias: str, tags: dict[str, str]) -> bool:
        """True if this candidate's AP >= the current @alias champion's AP (or there is none)."""
        try:
            new_ap = float(tags.get("average_precision", "nan"))
        except (TypeError, ValueError):
            new_ap = float("nan")
        if new_ap != new_ap:  # NaN: no measured AP -> never displace an incumbent
            return False
        try:
            incumbent = client.get_model_version_by_alias(name, alias)
        except Exception:  # noqa: BLE001, no incumbent (first champion) -> candidate wins
            return True
        try:
            inc_ap = float((incumbent.tags or {}).get("average_precision", "nan"))
        except (TypeError, ValueError):
            inc_ap = float("nan")
        # An incumbent without a measured AP should not block a measured candidate.
        return inc_ap != inc_ap or new_ap >= inc_ap

    def log_dataset(self, features: Any, source: str, name: str) -> None:
        """Record the training data as an MLflow dataset input, not a stringly param.

        `log_input` is the documented slot for "which data produced this run", it renders
        in the UI's dataset panel and is queryable, where a fingerprint param is only
        eyeball-comparable."""
        if not self.enabled or self._mlflow is None:
            return
        try:
            import numpy as np

            dataset = self._mlflow.data.from_numpy(
                np.asarray(features), source=source, name=name
            )
            self._safe(self._mlflow.log_input, dataset, context="training")
        except Exception as exc:  # noqa: BLE001
            _log.warning("log_dataset failed: %s: %s", type(exc).__name__, str(exc)[:100])

    def log_tags(self, tags: dict[str, str]) -> None:
        if not self.enabled or self._mlflow is None:
            return
        self._safe(self._mlflow.set_tags, {k: str(v)[:250] for k, v in tags.items()})

    def log_artifact_json(self, name: str, payload: Any) -> None:
        """Attach a JSON artifact, the full result, so a run is self-contained."""
        if not self.enabled or self._mlflow is None:
            return
        self._warn_if_artifacts_unreachable()
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_text(json.dumps(payload, indent=2, default=str))
            self._safe(self._mlflow.log_artifact, str(path))

    def _warn_if_artifacts_unreachable(self) -> None:
        """Loudly flag the one configuration where artifact logging silently drops: the SQLite
        FALLBACK path (server down) against experiments whose artifact_location is a server-only
        `mlflow-artifacts:` proxy. Runs and metrics still record to SQLite, but figures/models
        posted through the proxy have nowhere to land. We warn once so an artifact-producing run
        started while the server is down is never mistaken for one that saved its plots."""
        if getattr(self, "_artifact_warned", False):
            return
        if self.server_uri is None:  # SQLite fallback active
            _log.warning(
                "MLflow server is down: recording to the SQLite fallback. Metrics/params are "
                "saved, but artifacts (figures/models) may be dropped if the experiment's "
                "artifact_location is a server-only proxy. Start the MLflow server before an "
                "artifact-producing run (make mlflow-up)."
            )
        self._artifact_warned = True

    def log_figure(self, figure: Any, name: str) -> None:
        """Attach a matplotlib figure as a run artifact under `name` (e.g. `plots/pr.png`).

        The figure is closed after logging so a run that emits many plots does not leak
        matplotlib state. A logging failure is swallowed like every other telemetry op.
        """
        if not self.enabled or self._mlflow is None:
            self._close(figure)
            return
        self._warn_if_artifacts_unreachable()
        self._safe(self._mlflow.log_figure, figure, name)
        self._close(figure)

    @staticmethod
    def _close(figure: Any) -> None:
        try:
            import matplotlib.pyplot as plt

            plt.close(figure)
        except Exception:  # noqa: BLE001, closing is best-effort cleanup
            pass
