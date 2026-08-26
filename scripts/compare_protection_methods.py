"""The protection-technique frontier: tokenization vs anonymization vs synthetic data.

One corpus, three Protegrity capabilities, one measured trade-off. Tokenization preserves
per-row utility and leaves linkage structure intact (our attack suite quantifies the
residue). k-Anonymization generalizes quasi-identifiers until re-identification risk meets
a bound, at a measurable utility cost. Synthetic data severs row identity entirely, at
whatever fidelity the generator achieves. Each is the right tool for a different exposure;
this script measures all three on the same ledger so the choice is a number, not a vibe.

    python scripts/compare_protection_methods.py            # parts that need only the anon service
    python scripts/compare_protection_methods.py --with-synthetic

Requires the Developer Edition anonymization stack (vendor-de/anonymization,
localhost:8085); the synthetic part additionally needs vendor-de/synthetic-data.
Writes data/eval/protection_methods.json. No hosted-API or LLM calls; everything local.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

from amlguard import settings as _settings  # noqa: E402
from amlguard.persist import atomic_write_json  # noqa: E402

ANON_EP = _settings.anonymization_url()
CORPUS = ROOT / "data" / "corpus"
INTERVAL = re.compile(r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*[-,;]\s*(-?\d+(?:\.\d+)?)\s*[\)\]]")


def part1_metadata_risk(client) -> dict:
    """How re-identifiable is the metadata tokenization deliberately leaves clear?

    The architecture keeps jurisdiction, party type, risk rating, and PEP status
    unprotected so structured filtering works. That is a stated trade; this measures it
    with the vendor's own risk engine (prosecutor/journalist/marketer models).
    """
    parties = json.loads((CORPUS / "parties.json").read_text())
    qis = ["jurisdiction", "party_type", "risk_rating", "is_pep"]
    records = [{k: str(p.get(k, "")) for k in qis} for p in parties]
    risk = client.calculate_risk(data=records, quasi_identifiers=qis)
    return {
        "quasi_identifiers": qis,
        "n_records": len(records),
        "k_anonymity": risk.k_anonymity,
        "highest_risk_level": str(risk.highest_risk_level),
        "prosecutor_risk": round(float(risk.prosecutor.overall_risk), 4),
        "journalist_risk": round(float(risk.journalist.overall_risk), 4),
        # The third attacker model the risk engine computes: average-case re-identification
        # across the whole population, the natural framing for bulk-linkage exposure of the
        # left-clear KYC metadata. We computed the engine's full frontier but dropped
        # this one before.
        "marketer_risk": round(float(risk.marketer.overall_risk), 4)
        if getattr(risk, "marketer", None) is not None else None,
        "equivalence_classes": getattr(risk, "num_equivalence_classes", None),
    }


def _interval_midpoint(value: str) -> str:
    match = INTERVAL.search(str(value))
    if match:
        lo, hi = float(match.group(1)), float(match.group(2))
        return f"{(lo + hi) / 2:.2f}"
    return str(value)


def part2_generalized_amounts(client) -> dict:
    """AMOUNT under interval generalization, scored by the same classifier pipeline.

    Tokenized amounts scored AP 0.483 (90% of clear) because a token is a plausible but
    meaningless number. Generalization keeps magnitude, an interval midpoint is close to
    the truth, so the question is whether the k-anonymity toolset beats format-preserving
    tokenization on model utility. Encoding choice, stated: interval labels are mapped to
    their midpoints, the standard utility encoding for generalized numerics.
    """
    transactions = json.loads((CORPUS / "transactions.json").read_text())
    # The engine requires a sensitive attribute (the l-diversity target); channel is the
    # natural one here, what a row reveals about how value moved.
    # Carry a UNIQUE row key as an insensitive passthrough so we can join survivors back to
    # their original transaction EXACTLY, by id. The previous rebuild matched on `channel`
    # (11 distinct values across 6827 rows), so a suppressed row almost always shared its
    # channel with the next survivor and its amount was assigned to the WRONG transaction —
    # corrupting the committed _anon-monetary corpus and the reported frontier number. A
    # unique key removes the guesswork entirely.
    records = [
        {"row_uid": str(i), "amount": float(t["amount"]), "channel": str(t["channel"])}
        for i, t in enumerate(transactions)
    ]
    amounts = [r["amount"] for r in records]
    anonymized = client.anonymize(
        data=records,
        k=5,
        max_suppression=0.01,
        attributes=[
            {
                "name": "amount",
                "type": "quasi_identifier",
                "hierarchy": {
                    "type": "interval",
                    "params": {
                        "intervals": [1_000, 5_000, 25_000, 100_000],
                        "lower_bound": int(min(amounts)),
                        "upper_bound": int(max(amounts)) + 1,
                    },
                },
            },
            {"name": "channel", "type": "sensitive"},
            {"name": "row_uid", "type": "insensitive"},
        ],
    )
    anon_records = anonymized.data if hasattr(anonymized, "data") else anonymized
    # Capture the vendor's OWN quality metrics rather than only re-deriving suppression by hand:
    # the SDK returns information_loss and a suppressed_count on the result, which is the
    # authoritative measure of what k-anonymization cost this attribute.
    vendor_metrics = {}
    _metrics = getattr(anonymized, "metrics", None)
    if _metrics is not None:
        vendor_metrics = {
            "information_loss": getattr(_metrics, "information_loss", None),
            "suppressed_count": getattr(anonymized, "suppressed_count", None),
        }
    # Row parity by the unique key, NOT by value-matching a low-cardinality attribute and NOT
    # by trusting result order. Survivors carry their original row_uid (insensitive passthrough);
    # a suppressed row is any input id absent from the survivor set, and re-enters with an
    # unusable amount ('*'), the same semantics a redacted token has under protection. Building
    # the id->survivor map and walking the ORIGINAL order makes alignment exact and
    # order-independent, whether or not any rows were suppressed.
    by_uid = {str(r.get("row_uid")): r for r in anon_records}
    if len(by_uid) != len(anon_records):
        raise RuntimeError(
            "anonymizer did not preserve unique row_uid on survivors; cannot align safely")
    anon_records = [
        by_uid.get(rec["row_uid"],
                   {"amount": "*", "channel": rec["channel"], "row_uid": rec["row_uid"]})
        for rec in records
    ]

    variant_dir = ROOT / "data" / "protected" / "_anon-monetary"
    variant_dir.mkdir(parents=True, exist_ok=True)
    generalized = []
    for txn, anon in zip(transactions, anon_records):
        row = dict(txn)
        row["amount"] = _interval_midpoint(anon["amount"])
        generalized.append(row)
    atomic_write_json(variant_dir / "transactions.json", generalized)
    for name in ("narratives.json", "parties.json"):
        (variant_dir / name).write_text((ROOT / "data" / "protected" / "none" / name).read_text())

    from amlguard.training import train_scope

    result = train_scope(variant_dir, CORPUS, "anon-monetary")
    baseline = json.loads((ROOT / "data" / "eval" / "training.json").read_text())
    by_scope = {r["scope"]: r for r in baseline}
    return {
        "method": "k-anonymity interval generalization (k=5), midpoint-encoded",
        "average_precision": round(result.average_precision, 4),
        "roc_auc": round(result.roc_auc, 4),
        "tokenized_amount_ap": by_scope.get("direct-plus-monetary", {}).get("average_precision"),
        "clear_ap": by_scope.get("none", {}).get("average_precision"),
        # The vendor's own quality metric, not just our re-derived suppression count.
        "vendor_metrics": vendor_metrics,
    }


def part3_synthetic(with_synthetic: bool) -> dict | None:
    """A synthetic transaction table: identity severed by construction, fidelity measured.

    A vine copula models the joint distribution of transaction attributes and samples new
    rows. No real party id, transaction id, or date survives into the output, so row-level
    linkage is impossible by construction rather than by strength, the opposite end of the
    frontier from tokenization. What must be measured is fidelity: whether the synthetic
    table still looks like the ledger an analytics team would develop against. Scored with
    the vendor's own evaluate() plus distribution moments.
    """
    if not with_synthetic:
        return None
    import pandas as pd
    from synthetic_data_sdk import RemoteVineCopula
    from synthetic_data_sdk.config import ClientConfig

    transactions = json.loads((CORPUS / "transactions.json").read_text())
    df = pd.DataFrame(
        [{"amount": float(t["amount"]), "channel": str(t["channel"])} for t in transactions]
    )
    model = RemoteVineCopula(
        config=ClientConfig(endpoint=_settings.synthetic_url())
    )
    model.fit(df)
    synth = model.transform(len(df))

    metrics = {}
    try:
        metrics = model.evaluate(df, synth, categorical_cols=["channel"]) or {}
    except Exception as exc:  # noqa: BLE001, fidelity moments below still stand alone
        metrics = {"evaluate_error": f"{type(exc).__name__}: {exc}"}

    real_amt, synth_amt = df["amount"], synth["amount"].astype(float)
    real_ch = df["channel"].value_counts(normalize=True)
    synth_ch = synth["channel"].astype(str).value_counts(normalize=True)
    return {
        "rows": int(len(synth)),
        "identity_linkage": "none by construction: no real identifier is emitted",
        "amount_mean": {"real": round(real_amt.mean(), 2), "synthetic": round(synth_amt.mean(), 2)},
        "amount_p50": {"real": round(real_amt.median(), 2), "synthetic": round(synth_amt.median(), 2)},
        "amount_p95": {"real": round(real_amt.quantile(0.95), 2),
                        "synthetic": round(synth_amt.quantile(0.95), 2)},
        "channel_l1_distance": round(
            float(sum(abs(real_ch.get(c, 0) - synth_ch.get(c, 0))
                      for c in set(real_ch.index) | set(synth_ch.index))), 4),
        "vendor_evaluate": {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
        # TSTR: score the synthetic table by *task utility*, like the other frontier techniques
        # are scored (tokenization/k-anon go through train_scope), instead of fidelity moments
        # alone. Makes the frontier apples-to-apples.
        "tstr": _tstr(df, synth),
    }


def _tstr(real_df, synth_df) -> dict:
    """Train-on-Synthetic-Test-on-Real vs Train-on-Real-Test-on-Real, a downstream-task score.

    The task, learnable from the synthetic table's own columns: predict the channel from the
    amount (channels have distinct amount distributions). If the synthetic data preserved the
    joint distribution, a model trained on it should score on REAL data close to one trained on
    real data. The gap is the task-utility cost of synthesis, comparable to the AP-retention the
    other techniques report. Uses a fixed split and seed so the number is reproducible.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    real = real_df.copy()
    real["amount"] = real["amount"].astype(float)
    synth = synth_df.copy()
    synth["amount"] = synth["amount"].astype(float)
    synth["channel"] = synth["channel"].astype(str)

    # Real test set held out once; both models are tested on it.
    r_train, r_test = train_test_split(real, test_size=0.3, random_state=20260811)

    def fit_score(train_df):
        clf = RandomForestClassifier(n_estimators=80, random_state=0)
        clf.fit(train_df[["amount"]].to_numpy(), train_df["channel"].to_numpy())
        pred = clf.predict(r_test[["amount"]].to_numpy())
        return round(float(f1_score(r_test["channel"].to_numpy(), pred, average="macro")), 4)

    trtr = fit_score(r_train)          # train real -> test real (the reference)
    tstr = fit_score(synth)            # train synthetic -> test real
    # Chance macro-F1 for a k-class balanced problem is ~1/k. A retention ratio is only
    # meaningful when the reference (TRTR) is comfortably above chance; predicting channel from
    # amount alone is hard, so we flag whether the reference cleared chance and let the reader
    # discount the ratio if not, rather than presenting a ratio off a near-random baseline.
    n_classes = int(real["channel"].nunique())
    chance = round(1.0 / n_classes, 4) if n_classes else 0.0
    reference_above_chance = trtr is not None and trtr > 1.5 * chance
    return {
        "task": "predict channel from amount (macro-F1 on a held-out real test set)",
        "train_real_test_real": trtr,
        "train_synth_test_real": tstr,
        "utility_retained": round(tstr / trtr, 3) if trtr else None,
        "chance_macro_f1": chance,
        "reference_above_chance": reference_above_chance,
        "note": ("reference task is only marginally above chance, so the retention ratio is "
                 "weak evidence, read TSTR/TRTR as directional, not precise")
                if not reference_above_chance else "reference task is meaningfully above chance",
    }


def part4_differential_privacy(client) -> dict:
    """The fourth technique: differential privacy on an aggregate query.

    The design rules DP out for the RAG/serving path (an analyst sees the record verbatim) but
    names DP as legitimate for *aggregate artifacts* a bank would publish. A DP count of
    transactions by channel is exactly that: a typology-prevalence statistic released with a
    formal epsilon guarantee and no single transaction exposed. We ATTEMPT the real call and
    record the outcome honestly, no fabricated numbers.

    Measured: Developer Edition does not enable DP compute ("Differential privacy is not
    available on this tier"). So this frontier point is "the right technique for the
    shareable-aggregate stage, wired and attempted, gated by the DE tier", not a number. On a
    Team/Enterprise tier the same call would return a DP count per channel with its budget.
    """
    from collections import Counter

    txns = json.loads((ROOT / "data" / "protected" / "none" / "transactions.json").read_text())
    data = [{"channel": str(t.get("channel", "?"))} for t in txns[:2000]]
    true_counts = dict(Counter(d["channel"] for d in data))
    result: dict = {
        "technique": "differential privacy (DP count of transactions by channel, eps=1.0)",
        "stage": "shareable aggregate (the DP-appropriate stage)",
        "true_counts": true_counts,
    }
    try:
        from anonymization_sdk import DPMechanismType

        r = client.dp_compute(
            data=data, mechanism=DPMechanismType.COUNT, column="channel",
            group_by="channel", epsilon=1.0,
        )
        result["dp_counts"] = getattr(r, "results", None)
        result["budget"] = str(getattr(r, "budget", None))
        result["available"] = True
    except Exception as exc:  # noqa: BLE001, tier-gating is the honest finding
        result["available"] = False
        result["reason"] = str(exc)[:160]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-synthetic", action="store_true")
    parser.add_argument("--with-dp", action="store_true",
                        help="attempt the differential-privacy frontier point (tier-gated in DE)")
    args = parser.parse_args()

    try:
        from anonymization_sdk import AnonymizationClient
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"anonymization_sdk not importable: {exc}")

    client = AnonymizationClient(base_url=ANON_EP)

    # Preflight the local anonymization service so a stopped container fails with a start
    # instruction, not a raw connection traceback mid-run (matches healthcare_deidentify.py).
    try:
        out = {
            "metadata_risk": part1_metadata_risk(client),
            "generalized_amounts": part2_generalized_amounts(client),
        }
    except Exception as exc:  # noqa: BLE001, the local service being down is the likely cause
        sys.exit(
            f"Anonymization service call failed: {type(exc).__name__}: {exc}\n"
            f"Start it with: cd vendor-de/anonymization && docker compose up -d "
            f"(expected at {ANON_EP})."
        )
    synth = part3_synthetic(args.with_synthetic)
    if synth is not None:
        out["synthetic_ledger"] = synth
    if args.with_dp:
        out["differential_privacy"] = part4_differential_privacy(client)

    dest = ROOT / "data" / "eval" / "protection_methods.json"
    atomic_write_json(dest, out)
    print(json.dumps(out, indent=2))
    print(f"\nwritten to {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
