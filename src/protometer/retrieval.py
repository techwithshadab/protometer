"""Retrieval over protected text, where Semantic Erasure becomes measurable.

Narratives are embedded and indexed **after** tokenization, so the vector store never holds
a real identifier. Retrieval returns token-bearing chunks, and those chunks are what reach
the LLM.

The consequence is the phenomenon this project exists to measure. A token such as
`4oB93 T7MdI3` carries no embedding relationship to the `Leila Rahman` it replaced, so
similarity search over the protected value itself is not degraded but **destroyed**.
Retrieval still works to the extent that the *surrounding unprotected prose* carries the
signal, "cash deposits below the reporting threshold" still embeds meaningfully. Widening
[[Protection Scope]] progressively removes that surrounding signal too, which is why
retrieval quality is expected to fall as scope widens, by a different mechanism than the one
degrading the LLM's reasoning.

Metadata is deliberately kept unprotected: party ids, risk ratings, jurisdictions and
document types are non-identifying, and keeping them queryable demonstrates that structured
filtering survives protection intact even when semantic search over identifiers does not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

# Sentence-transformers all-MiniLM-L6-v2 via Chroma's default embedding function. Chosen
# because it runs locally: no narrative text, protected or otherwise, leaves the machine
# during indexing.
COLLECTION_PREFIX = "protometer"


class StaleIndexError(RuntimeError):
    """Raised when a persisted index does not match the corpus being evaluated."""


class EmptyIndexError(RuntimeError):
    """Raised when searching an index that was never built.

    Chroma's `get_or_create_collection` creates a missing collection rather than failing, and
    `NarrativeIndex.__init__` calls `mkdir(parents=True, exist_ok=True)`, so on a clone with
    no `data/index/` (which is gitignored) every search silently returned `[]`. Every task
    then ran with "(no case notes retrieved)" and still scored, and
    `measure_semantic_erasure.py` scored the empty behavioural arm as distance **0.0**, the
    best value on a lower-is-better scale. A missing index therefore reproduced this
    project's headline retrieval finding exactly, with no index at all.

    Zero results is a legitimate answer to a *query*; zero documents in the *index* is a
    build failure, and the two must not be indistinguishable.
    """


def corpus_fingerprint(narratives: list[dict]) -> str:
    """Stable digest of a narrative corpus, for the stale-index guard.

    Single definition, because the builder (`build_index`) and the guard
    (`NarrativeIndex.index_narratives`) must compute the *same* value or the guard silently
    passes a stale index: if the two expressions ever diverged, a regenerated corpus would be
    served against the previous corpus's chunks with no error. They were duplicated inline;
    this is the one place the expression lives.

    Hashes the full TEXT content, not its length: format-preserving tokenization (and most
    corpus edits) swap a string for a DIFFERENT string of the SAME length, so a length-only
    digest collided and let a superseded index pass the guard — exactly the stale-served failure
    this exists to prevent. A newline delimiter (absent from document_id) keeps field boundaries
    unambiguous so two documents cannot alias by concatenation.
    """
    digest = hashlib.sha256()
    for n in narratives:
        digest.update(f"{n['document_id']}\n".encode())
        digest.update(f"{n['text']}\n".encode())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: str
    text: str
    metadata: dict[str, Any]
    distance: float

    @property
    def subject_party_id(self) -> str:
        return str(self.metadata.get("subject_party_id", ""))


class NarrativeIndex:
    """A vector index over one protected corpus.

    One index per [[Protection Scope]], they must never share a collection, since the whole
    point is that the same query retrieves differently depending on how much of the text was
    tokenized before embedding.
    """

    def __init__(
        self, scope_slug: str, persist_dir: Path, corpus_fingerprint: str = ""
    ) -> None:
        self.scope_slug = scope_slug
        self.corpus_fingerprint = corpus_fingerprint
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        # The fingerprint is set at creation because Chroma forbids modifying a collection's
        # distance function afterwards, and `modify` replaces the whole metadata dict.
        metadata = {"hnsw:space": "cosine"}
        if corpus_fingerprint:
            metadata["corpus_fingerprint"] = corpus_fingerprint
        self._collection = self._client.get_or_create_collection(
            name=f"{COLLECTION_PREFIX}_{scope_slug.replace('-', '_')}",
            metadata=metadata,
        )

    def __len__(self) -> int:
        return self._collection.count()

    def validate_fingerprint(self) -> None:
        """Raise StaleIndexError if this non-empty index was built from a DIFFERENT corpus.

        The build path (`index_narratives`) checks staleness, but the eval READ path constructs
        the index and calls `search()` directly, which only guards against an EMPTY index. A
        non-empty index left over from a superseded corpus would then be queried silently,
        contaminating every task's retrieved notes. Callers on the read path must call this once
        after construction (with `corpus_fingerprint` supplied) to enforce the same guard the
        build path has. No-op when this index carries no expected fingerprint or is empty.
        """
        if not self.corpus_fingerprint or len(self) == 0:
            return
        stored = (self._collection.metadata or {}).get("corpus_fingerprint")
        if stored != self.corpus_fingerprint:
            raise StaleIndexError(
                f"Index for scope {self.scope_slug!r} was built from a different corpus "
                f"(stored {stored}, current {self.corpus_fingerprint}). Delete data/index/"
                f"{self.scope_slug} and rebuild with: python scripts/ingest_all.py {self.scope_slug}"
            )

    def index_narratives(self, narratives: list[dict], batch_size: int = 256) -> int:
        """Embed and store protected narratives.

        Narratives are indexed whole rather than chunked: each is a single investigator note
        of a few hundred characters, already the natural unit of retrieval. Chunking would
        split entity mentions from the context that gives them meaning, which would confound
        the measurement with an artefact of the chunking strategy.
        """
        # An index is reused only when it was built from *this* corpus. Returning early on
        # `len(self) > 0` alone silently served a stale index after the corpus was regenerated
        #, retrieval would answer the new evaluation with the previous corpus's chunks, with
        # no error and a plausible-looking count.
        fingerprint = self.corpus_fingerprint or corpus_fingerprint(narratives)
        stored = (self._collection.metadata or {}).get("corpus_fingerprint")

        if len(self) > 0:
            if stored == fingerprint:
                return len(self)
            raise StaleIndexError(
                f"Index for scope {self.scope_slug!r} was built from a different corpus "
                f"(stored {stored}, current {fingerprint}). Delete data/index/"
                f"{self.scope_slug} and rebuild."
            )

        ids = [n["document_id"] for n in narratives]
        documents = [n["text"] for n in narratives]
        metadatas = [
            {
                "document_id": n["document_id"],
                "document_type": n.get("document_type", ""),
                "subject_party_id": n.get("subject_party_id", ""),
                # Present only on typology narratives; empty string keeps Chroma's metadata
                # types uniform. Never used for retrieval, it would leak the answer.
                "typology_id": n.get("typology_id") or "",
            }
            for n in narratives
        ]

        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            self._collection.add(
                ids=ids[start:stop],
                documents=documents[start:stop],
                metadatas=metadatas[start:stop],
            )

        return len(self)

    def search(
        self, query: str, top_k: int = 6, where: dict | None = None
    ) -> list[RetrievedChunk]:
        """Semantic search over protected text.

        The query arrives in clear language ("cash deposits below the reporting threshold"),
        while the index holds tokenized text. That asymmetry is the point: queries about
        *behaviour* still match, and queries about *identity* cannot.

        Raises [[EmptyIndexError]] rather than returning `[]` when nothing was ever indexed:
        an unbuilt index otherwise produces a plausible-looking null result instead of an
        error, which is how a missing build step came to reproduce the retrieval finding.
        """
        if len(self) == 0:
            raise EmptyIndexError(
                f"Index for scope {self.scope_slug!r} is empty, nothing was indexed. "
                f"Build it with: python scripts/ingest_all.py {self.scope_slug}"
            )

        result = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, max(len(self), 1)),
            where=where or None,
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        return [
            RetrievedChunk(
                document_id=str(doc_id),
                text=text,
                metadata=dict(metadata or {}),
                distance=float(distance),
            )
            for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances)
        ]


def build_index(protected_dir: Path, index_root: Path, scope_slug: str) -> NarrativeIndex:
    """Build (or open) the index for one protected corpus."""
    narratives = json.loads((protected_dir / "narratives.json").read_text())
    fingerprint = corpus_fingerprint(narratives)
    index = NarrativeIndex(scope_slug, index_root / scope_slug, fingerprint)
    index.index_narratives(narratives)
    return index
