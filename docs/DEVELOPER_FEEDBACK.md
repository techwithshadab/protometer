# Developer feedback: building on Protegrity Developer Edition

Optional feedback for the Protegrity team. Everything below is *measured*, with a reproduction
script in the repo; the platform behaviours are catalogued in
[protegrity-api-reference.md](protegrity-api-reference.md).

## What worked well

- **Deterministic tokenization** (undocumented, verified across four data elements) is the
  foundation of our batching: one full 8-scope ingest is 14 API calls instead of ~2,687, with
  byte-identical re-protection verified live. This one property shaped the whole architecture.
- The **batch protect / unprotect / reprotect** API is fast and dependable once you know the
  success codes. `reprotect` performing server-side key rotation with plaintext never transiting
  the application is a genuinely nice primitive; we demo it live.
- The **anonymization SDK** (Safe Harbor + the Expert-Determination risk engine) made a real
  HIPAA de-identification measurement possible with modest code.

## What tripped us up (each measured, each with a repro)

- **Protection can fail silently.** `protect("seven hundred twelve thousand dollars", "number")`
  returns the input unchanged *with a success code*. Emitting plaintext under a protection claim
  is a correctness failure, not a trade-off, so we compare every token against its input and
  redact no-ops. A distinct "not-protected" signal in the response would remove a whole class of
  silent leaks. (`scripts/format_coverage.py`)
- **`ORGANIZATION` detection returns zero** at every threshold from 0.6 down to 0.0, though
  `ORGANIZATION` is in the SDK's own element mapping. Fatal for a corpus that is 45%
  organizations; we had to build a deterministic known-entity roster to close it.
- **Detection is format-sensitive.** 3 of 20 probed formats missed entirely (prose dates,
  European decimals, written-out amounts); identifiers were robust across all variants.
- **The burst limit returns `403`, not `429`,** and blocks for ~4 minutes. The SDK retries on
  `429` only, so its own backoff never engages against the actual limit. Distinct, documented
  error semantics for rate-limiting vs. auth failure would help a lot here.
- **`/auth/login` is rate-limited separately from `/protect`,** and the SDK surfaces that `429`
  as a *credential* failure. This cost us a debugging afternoon; we now share one session per
  scope so logins stay to roughly one per scope.
- **Batch success code `50` (reprotect) is undocumented** (protect/unprotect are 6/8). We pinned
  it with a regression test.

## What we'd love to see

1. Documented **determinism guarantees** (it is load-bearing for anyone batching).
2. Distinct **error semantics** for rate-limiting vs. authentication failure.
3. **Organization-name detection**, or guidance that a customer master is expected to backstop it.
4. A **"was this actually protected?"** flag on the protect response, so silent no-ops surface
   without caller-side value comparison.

Thank you for a genuinely well-designed challenge and for the free Developer Edition access that
made an evaluation this thorough affordable.
