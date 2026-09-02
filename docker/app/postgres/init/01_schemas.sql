-- Protometer Postgres bootstrap. Runs once on an empty data dir.
--
-- Per-domain schemas keep the corpus tables namespaced by domain (the same domain seam the code
-- uses): the AML corpus lives in `aml.*`, and healthcare/support get their own schemas so the
-- same loader populates them when they have data.
--
-- MLflow deliberately does NOT use this Postgres (ADR-0053): its model-registry alias API is
-- broken against Postgres in the pinned MLflow version, so MLflow's backend stays on its SQLite
-- store where the champion registry works. Only the corpus mirror lives here.
--
-- Table DDL is NOT here: the loader (scripts/load_corpus_db.py) creates corpus tables from the
-- JSON shape so the schema follows the corpus, not a hand-maintained copy that could drift. This
-- file only guarantees the schemas exist.

CREATE SCHEMA IF NOT EXISTS aml;
CREATE SCHEMA IF NOT EXISTS healthcare;
CREATE SCHEMA IF NOT EXISTS support;

-- The single app role owns everything; grants are explicit so a future read-only role is easy.
GRANT ALL ON SCHEMA aml, healthcare, support TO protometer;
ALTER ROLE protometer SET search_path = aml, public;
