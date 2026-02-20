"""Summarisation Agent (Tier 0).

Produces a focused summary of the candidate application.
Supports parameterised focus — the requesting agent specifies what
aspects to emphasise.

Uses the LLMClient protocol — swappable between simulated (for testing)
and real (via LiteLLM) implementations.
"""

from __future__ import annotations

import typing as t
from datetime import datetime, timezone

from hiring_agents.models import (
    AgentDecisionRecord,
    ReasoningStep,
    SummarisationOutput,
)
from hiring_agents.state import TransactionState

AGENT_ID = "summariser:001"


class LLMClient(t.Protocol):
    """Protocol for LLM interaction — allows swapping real/simulated."""

    def summarise(self, text: str, focus: str) -> tuple[str, list[str]]:
        """Return (summary, key_points)."""
        ...

    def synthesise_reasoning(
        self,
        criteria_summary: str,
        application_summary: str,
        match: str,
        action: str,
    ) -> str:
        """Produce a reasoning narrative for a screening decision."""
        ...


def run_summariser(
    state: TransactionState,
    focus: str = "screening",
    llm_client: LLMClient | None = None,
) -> SummarisationOutput:
    """Summarise the application with the specified focus.

    The focus parameter is set by the requesting agent:
    - "screening" — for the Screening Agent (highlight skills match, experience)
    - "deep_assessment" — for the Candidate Profile Agent (highlight gaps, inconsistencies)
    - "interview_prep" — for the Interview Recommendation Agent (highlight areas to probe)
    """
    if llm_client is None:
        from hiring_agents.llm import SimulatedLLMClient

        llm_client = SimulatedLLMClient()

    app = state.application
    all_content = "\n\n".join(
        f"[{doc.document_type}] {doc.content}" for doc in app.documents
    )

    summary, key_points = llm_client.summarise(all_content, focus)

    output = SummarisationOutput(
        application_id=app.application_id,
        summary=summary,
        focus=focus,
        key_points=key_points,
    )

    state.record_decision(AgentDecisionRecord(
        agent_id=AGENT_ID,
        timestamp=datetime.now(timezone.utc),
        inputs_summary={
            "application_id": app.application_id,
            "focus": focus,
            "content_length": len(all_content),
        },
        reasoning_steps=[
            ReasoningStep(
                step="summarise",
                detail=f"Produced {focus}-focused summary of {len(all_content)} chars",
            ),
        ],
        output=output,
        confidence=0.85,
        model_id=type(llm_client).__name__,
    ))

    state.summarisation = output
    return output
