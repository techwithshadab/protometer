"""Test-session hygiene: telemetry off, so test fixtures never appear in real dashboards."""

import os

# Fake-model fixtures exercising the LLM client must not export spans to a live Langfuse
# or runs to a live MLflow, a test that pollutes production observability is a test with
# a side channel.
os.environ.setdefault("AMLGUARD_NO_TRACING", "1")
os.environ.setdefault("AMLGUARD_NO_TRACKING", "1")
