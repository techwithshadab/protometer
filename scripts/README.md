# Scripts — the command palette

Each script is a thin CLI entry point over the `protometer` library in [`src/`](../src/protometer/) — it
parses args, loads `.env`, and calls the library. **No pipeline logic lives here** (see the library),
and there is no duplication between the two. Grouped by what you'd run them for:

### Build the corpus & protect it
| Script | Does |
|---|---|
| `build_corpus.py` | Generate the synthetic AML corpus into `data/corpus/`. |
| `ingest_all.py` | Protect the corpus under every protection scope (unattended). |
| `load_corpus_db.py` | Load the corpus JSON into Postgres (the app's source of truth). |
| `check_determinism.py` | Prove tokenization is deterministic before anything else. |
| `verify_protection.py` | Leak check: no in-scope identifier survives in a protected corpus. |
| `verify_corpus_parity.py` | Confirm every protected corpus derives from the same generation. |

### Run the measurements (the instrument)
| Script | Does |
|---|---|
| `run_eval.py` | The utility-vs-scope evaluation (the headline curve). |
| `run_training.py` | Train a classifier per scope; report the utility-vs-scope cost. |
| `run_hybrid.py` | The hybrid: classifier ranks the queue, the model reasons over the head. |
| `run_attacks.py` | The adversarial evaluation (what protection does *not* protect). |
| `measure_semantic_erasure.py` | The headline retrieval-asymmetry finding. |
| `run_statistics.py` | Confidence intervals, paired tests, power for the curve. |
| `estimate_cost.py` | Estimate an eval run's cost **before** spending. |
| `compare_models.py` / `compare_protection_methods.py` | Cross-model / cross-technique comparisons. |
| `healthcare_deidentify.py` | HIPAA de-identification, measured (Safe Harbor + Expert Determination). |
| `format_coverage.py` | Which value formats the discovery service actually detects. |

### Run the product (demos & serving)
| Script | Does |
|---|---|
| `demo.py` | One command, the whole pipeline end to end. |
| `demo_chat.py` | Multi-turn protected chatbot demo (any domain, live). |
| `demo_serving.py` | The serving boundary on a chatbot + a batch queue. |
| `demo_support_gates.py` | The customer-support role dual-gate. |
| `setup_local_model.py` | One-time pull of the open-source local model (or `make setup-local-model`). |

### Observability & governance
| Script | Does |
|---|---|
| `govern_models.py` | Reconcile the MLflow model registry (champion/archived aliases). |
| `observability_report.py` | Join MLflow + Langfuse + Prometheus into one consolidated view. |

### Generate & check the docs (keep numbers honest)
| Script | Does |
|---|---|
| `key_metrics.py` | Single source of truth for every headline number the docs cite. |
| `generate_results.py` | Regenerate `docs/results-<domain>.md` from evaluation output. |
| `generate_summary.py` | Regenerate the README's cross-domain summary. |
| `check_docs.py` | Fail if a hand-written doc cites a number that disagrees with the artifacts. |
| `build_journey.py` / `regen_one_turn.py` | Regenerate the UI's committed replay artifacts. |

Most have `--help`. The common ones are also wrapped as `make` targets — see the [Makefile](../Makefile)
and [docs/SETUP.md](../docs/SETUP.md).
