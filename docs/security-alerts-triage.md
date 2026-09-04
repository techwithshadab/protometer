# Security alert triage — 2026-09-03

Read-only assessment of the 6 open Dependabot alerts on `main` (`cf435ed`). No changes made.
Reachability was checked against the actual code and deploy topology, not just the dependency graph.

## Summary

| Package | Sev | Advisory | Vulnerable path | Reachable here? | Patch exists? |
|---|---|---|---|---|---|
| chromadb | critical | GHSA-f4j7-r4q5-qw2c | pre-auth code injection (server) | **No** — embedded | none |
| chromadb | critical | GHSA-36p7-vc44-83pf | code injection (server) | **No** — embedded | none |
| chromadb | high | GHSA-xph7-9rjv-w5fr | RBAC tenant check (server) | **No** — embedded | none |
| chromadb | high | GHSA-2wm9-hf6c-p5cr | authenticated arbitrary R/W (server) | **No** — embedded | none |
| mlflow | high | GHSA-h7x2-h6g9-p789 | AI Gateway SSRF (gateway server) | **No** — tracking client only | none (≤3.15.2 affected) |
| cryptography | high | GHSA-g6cj-pr64-35w5 | PKCS#7 EnvelopedData decrypt (CVE-2026-69247) | **No** — PKCS#7 API unused | **50.0.0** |

**Bottom line:** all 6 are present in the graph but none of the vulnerable code paths are exercised in
this deployment. None is a live exposure. Only one has a released fix (cryptography 50.0.0), and it is
**blocked** by mlflow's `cryptography<50` constraint.

## Details + reasoning

### chromadb (2 critical, 2 high) — already documented as not-affected
- Direct dep (`pyproject.toml:30`), imported in `src/protometer/retrieval.py` as
  `chromadb.PersistentClient` (embedded, in-process). No Chroma HTTP server is ever started; grep for a
  server entrypoint finds none; botox uses Neo4j, not Chroma.
- All four advisories are **server-mode / auth-side** (HTTP `/api/v2`, RBAC, server-side model loading).
  An embedded client exposes none of them. No patched release exists in any case.
- **This is already recorded** in `README.md:623-629` ("Known chromadb advisories do not apply…").
  No action needed; the alert can be dismissed as not-affected on GitHub, which the README already
  justifies.

### mlflow (high, AI Gateway SSRF) — not reachable
- Only the **tracking client** is used (`src/protometer/tracking.py`: `MlflowClient`,
  `set_tracking_uri` → a local SQLite store). The CVE is in the **MLflow AI Gateway / deployments
  server**, which this project never runs.
- **No patched version released** (the whole ≥3.13.0 line through your 3.15.2 pin is affected). There is
  nothing to upgrade *to* even if you wanted to.
- Same class as chromadb: dismiss as not-affected (path not run), and revisit if you ever stand up the
  MLflow gateway.

### cryptography (high, PKCS#7) — has a fix, but blocked, and the path is unused
- Transitive only (pulled by `mlflow` and `google-auth`, for TLS). No `src/` code imports it; grep for
  `pkcs7`/`EnvelopedData` finds **nothing**, so the vulnerable decryption API is never called.
- The fix (50.0.0) is **unreachable**: `mlflow` requires `cryptography<50,>=43.0.0`. `uv lock
  --upgrade-package cryptography` resolves right back to **49.0.0** (verified). It cannot move to 50
  until mlflow relaxes that pin — and mlflow itself has no patched release yet.

## The one concrete inconsistency to fix (low urgency, safe)

PR #4 (merged) bumped **only the generated `requirements.txt` to `cryptography==50.0.0`**, but:
- `uv.lock` (the real source of truth) stays at **49.0.0**, correctly, because of mlflow's `<50`.
- CI and the Docker images install via `uv sync --frozen` (from `uv.lock`), so they get **49.0.0** — the
  50.0.0 in `requirements.txt` is never installed and is not even a resolvable version given the
  constraints.

So `requirements.txt` now claims a version the graph forbids. The fix is to **regenerate
`requirements.txt` from the lockfile** so the two agree (both 49.0.0):

```bash
uv export --frozen --no-dev --no-emit-project > requirements.txt
# review: cryptography line goes 50.0.0 -> 49.0.0, matching uv.lock
git add requirements.txt
git commit -m "deps: resync requirements.txt to uv.lock (cryptography 49.0.0; 50 blocked by mlflow<50)"
git push
```

This is honest (it reflects what actually installs) and closes the file mismatch. It does **not** leave
you more exposed — the vulnerable cryptography path is unused either way.

## Recommended actions, prioritized

1. **Now (safe, 1 command):** regenerate `requirements.txt` from `uv.lock` (above) so the two stop
   disagreeing. Owner runs the commit/push.
2. **Now (GitHub, no code):** dismiss the 4 chromadb + 1 mlflow + 1 cryptography alerts as
   **"not affected — vulnerable path not reached"**, citing this file / `README.md:623`. Keep the
   dismissal note so the reasoning is on record (the repo's stated policy is to say so, not hide it).
3. **Optional (document parity):** add a short "cryptography & mlflow advisories: not-affected" bullet to
   `README.md`'s Honest-limitations section, mirroring the chromadb one, so all six are covered in prose.
4. **Watch:** when mlflow ships a release that both fixes its own SSRF CVE **and** allows
   `cryptography>=50`, bump mlflow (re-verifying `docs/results-*.md` still reproduce, since the ML stack
   is pinned for reproducibility) — that single bump clears the mlflow and cryptography alerts together.

## What NOT to do

- Don't `uv lock --upgrade-package cryptography` expecting 50.0.0 — it can't reach it under mlflow's
  pin; you'd just re-lock at 49.0.0 and think you fixed it.
- Don't hand-edit `requirements.txt` to a version `uv.lock` disallows — the two must stay in sync, and
  `uv.lock` wins.
- Don't bump mlflow off its pin casually — CLAUDE.md pins the ML measurement stack for reproducibility
  of committed results; any bump must re-verify `docs/results-*.md`.
