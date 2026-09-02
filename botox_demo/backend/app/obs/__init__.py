"""Observability for the BOTOX assistant: agent tracing to Langfuse.

The public surface is `tracing.trace_turn(...)` (a context manager for one chat turn) and
`tracing.span(...)` (a nested step). Both degrade to no-ops when Langfuse is not configured, so
the pipeline runs identically with or without observability.
"""
