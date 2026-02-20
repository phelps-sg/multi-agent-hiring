"""Document Classification Agent (Tier 0).

Classifies documents in a candidate application by type.
In the PoC, this uses simple heuristics. In production, this would
be an LLM-based classifier with confidence gating.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hiring_agents.models import (
    AgentDecisionRecord,
    CandidateApplication,
    ClassificationOutput,
    ReasoningStep,
    RoleRequirements,
)
from hiring_agents.state import TransactionState

AGENT_ID = "classifier:001"

# Simple keyword-based classification for the PoC
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "cv": ["cv", "curriculum vitae", "resume", "résumé", "work experience", "employment history"],
    "cover_letter": ["cover letter", "dear hiring", "i am writing to apply", "i am applying"],
    "portfolio": ["portfolio", "project samples", "work samples", "github"],
    "references": ["references", "referee", "recommendation letter"],
}


def _classify_document(content: str) -> str:
    content_lower = content.lower()
    for doc_type, keywords in _TYPE_KEYWORDS.items():
        if any(kw in content_lower for kw in keywords):
            return doc_type
    return "unknown"


def run_classifier(state: TransactionState) -> ClassificationOutput:
    """Classify all documents in the application and write results to state."""
    app = state.application
    reqs = state.role_requirements

    classifications: dict[str, str] = {}
    for doc in app.documents:
        # Use declared type if available, otherwise classify from content
        if doc.document_type and doc.document_type != "unknown":
            classifications[doc.document_id] = doc.document_type
        else:
            classifications[doc.document_id] = _classify_document(doc.content)

    documents_present = list(set(classifications.values()) - {"unknown"})
    documents_missing = [
        req_doc for req_doc in reqs.required_documents
        if req_doc not in documents_present
    ]

    output = ClassificationOutput(
        application_id=app.application_id,
        document_classifications=classifications,
        documents_present=documents_present,
        documents_missing=documents_missing,
    )

    # Record decision for audit trail
    state.record_decision(AgentDecisionRecord(
        agent_id=AGENT_ID,
        timestamp=datetime.now(timezone.utc),
        inputs_summary={
            "application_id": app.application_id,
            "document_count": len(app.documents),
            "required_documents": reqs.required_documents,
        },
        reasoning_steps=[
            ReasoningStep(
                step="classify_documents",
                detail=f"Classified {len(app.documents)} documents: {classifications}",
            ),
            ReasoningStep(
                step="check_completeness",
                detail=f"Present: {documents_present}; Missing: {documents_missing}",
            ),
        ],
        output=output,
        confidence=0.9,  # heuristic classifier, fixed confidence
        model_id="heuristic",
    ))

    state.classification = output
    return output
