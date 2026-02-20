"""Screening Agent (Tier 1 — analogous to Triage Agent).

Orchestrates sub-agents through shared state and evaluates the candidate
against the role requirements. Produces a structured ScreeningOutput with
full reasoning trace.

Key alignment properties:
- Evaluates only criteria defined in RoleRequirements (cannot invent criteria)
- Criterion evaluation is deterministic (no LLM involvement)
- All sub-agent interactions flow through observable TransactionState
- Produces AgentDecisionRecord for audit trail
- Recommend-only: never makes binding decisions
"""

from __future__ import annotations

from datetime import datetime, timezone

from hiring_agents.evaluate import evaluate_all_criteria
from hiring_agents.models import (
    AgentDecisionRecord,
    ReasoningStep,
    RiskFlag,
    ScreeningOutput,
)
from hiring_agents.state import TransactionState
from hiring_agents.sub_agents.classifier import run_classifier
from hiring_agents.sub_agents.summariser import LLMClient, run_summariser

AGENT_ID = "screening:001"


def _derive_match(
    criteria: list,
    flags: list[RiskFlag],
) -> tuple[str, float, str, str]:
    """Derive match level, confidence, priority, and recommended action
    from evaluated criteria and flags.

    Returns (match, confidence, priority, recommended_action).
    """
    total = len(criteria)
    if total == 0:
        return "possible-match", 0.5, "standard", "refer-to-hiring-manager"

    passes = sum(1 for c in criteria if c.status == "pass")
    fails = sum(1 for c in criteria if c.status == "fail")
    unknowns = sum(1 for c in criteria if c.status == "unknown")

    # Any high-severity flag or exclusion trigger fail → no-match
    has_exclusion_fail = any(
        c.criterion.startswith("exclusion_trigger:") and c.status == "fail"
        for c in criteria
    )
    has_high_flag = any(f.severity == "high" for f in flags)

    if has_exclusion_fail or has_high_flag:
        return "no-match", 0.95, "low", "reject"

    pass_rate = passes / total
    fail_rate = fails / total
    unknown_rate = unknowns / total

    if fail_rate > 0.3:
        confidence = min(0.9, 0.5 + fail_rate)
        return "no-match", confidence, "low", "reject"

    if unknown_rate > 0.3:
        confidence = max(0.3, 0.5 - unknown_rate)
        return "possible-match", confidence, "standard", "request-info"

    if pass_rate >= 0.8 and fails == 0:
        confidence = min(0.95, 0.6 + pass_rate * 0.3)
        return "strong-match", confidence, "urgent", "advance"

    if pass_rate >= 0.6:
        confidence = min(0.8, 0.5 + pass_rate * 0.2)
        return "possible-match", confidence, "standard", "refer-to-hiring-manager"

    confidence = max(0.4, pass_rate)
    return "possible-match", confidence, "standard", "refer-to-hiring-manager"


def _detect_flags(state: TransactionState) -> list[RiskFlag]:
    """Detect risk flags from the application data.

    These are heuristic checks for common red flags in applications.
    """
    flags: list[RiskFlag] = []
    app = state.application

    # Check for very short documents (potential low-effort application)
    for doc in app.documents:
        if len(doc.content.split()) < 20:
            flags.append(RiskFlag(
                flag="very_short_document",
                severity="medium",
                detail=f"Document '{doc.document_id}' has fewer than 20 words",
            ))

    # Check for salary expectation far above band
    reqs = state.role_requirements
    if (
        app.salary_expectation is not None
        and reqs.salary_band_max is not None
        and app.salary_expectation > reqs.salary_band_max * 1.2
    ):
        flags.append(RiskFlag(
            flag="salary_significantly_above_band",
            severity="high",
            detail=(
                f"Candidate expects {app.salary_expectation}; "
                f"band max is {reqs.salary_band_max} (>20% above)"
            ),
        ))

    return flags


def run_screening(
    state: TransactionState,
    llm_client: LLMClient | None = None,
) -> ScreeningOutput:
    """Run the full screening pipeline.

    1. Invoke Classification Agent → shared state
    2. Invoke Summarisation Agent → shared state
    3. Evaluate all criteria deterministically
    4. Detect risk flags
    5. Derive match/action/priority
    6. Record decision
    """
    invoked: list[str] = []

    # Step 1: Classification
    run_classifier(state)
    invoked.append("classifier:001")

    # Step 2: Summarisation (with screening focus)
    run_summariser(state, focus="screening", llm_client=llm_client)
    invoked.append("summariser:001")

    # Step 3: Evaluate criteria
    criteria = evaluate_all_criteria(
        state.application,
        state.role_requirements,
        state.classification,
    )

    # Step 4: Detect flags
    flags = _detect_flags(state)

    # Step 5: Derive outcome
    match, confidence, priority, action = _derive_match(criteria, flags)

    # Gather missing information
    missing = [c.criterion for c in criteria if c.status == "unknown"]
    if state.classification and state.classification.documents_missing:
        missing.extend(
            f"document:{d}" for d in state.classification.documents_missing
        )

    # Build reasoning — use LLM synthesis if available, else deterministic
    passes = [c for c in criteria if c.status == "pass"]
    fails = [c for c in criteria if c.status == "fail"]
    unknowns = [c for c in criteria if c.status == "unknown"]

    criteria_summary = "\n".join(
        f"  {c.criterion}: {c.status} — {c.detail}" for c in criteria
    )
    app_summary = state.summarisation.summary if state.summarisation else ""

    if llm_client is not None and hasattr(llm_client, "synthesise_reasoning"):
        reasoning = llm_client.synthesise_reasoning(
            criteria_summary=criteria_summary,
            application_summary=app_summary,
            match=match,
            action=action,
        )
    else:
        reasoning = (
            f"Evaluated {len(criteria)} criteria: "
            f"{len(passes)} pass, {len(fails)} fail, {len(unknowns)} unknown. "
            f"{len(flags)} risk flags detected. "
            f"Recommendation: {action} (confidence: {confidence:.2f})."
        )

    output = ScreeningOutput(
        application_id=state.application.application_id,
        match=match,
        confidence=confidence,
        priority=priority,
        recommended_action=action,
        missing_information=missing,
        flags=flags,
        reasoning=reasoning,
        criteria_evaluated=criteria,
    )

    # Step 6: Record decision
    state.record_decision(AgentDecisionRecord(
        agent_id=AGENT_ID,
        timestamp=datetime.now(timezone.utc),
        inputs_summary={
            "application_id": state.application.application_id,
            "role": state.role_requirements.role_title,
            "criteria_count": len(criteria),
            "documents_classified": len(state.application.documents),
        },
        reasoning_steps=[
            ReasoningStep(step="classify_documents", detail=f"Invoked {invoked[0]}"),
            ReasoningStep(step="summarise_application", detail=f"Invoked {invoked[1]} with focus=screening"),
            ReasoningStep(step="evaluate_criteria", detail=f"{len(passes)} pass, {len(fails)} fail, {len(unknowns)} unknown"),
            ReasoningStep(step="detect_flags", detail=f"{len(flags)} flags detected"),
            ReasoningStep(step="derive_recommendation", detail=f"match={match}, action={action}, confidence={confidence:.2f}"),
        ],
        output=output,
        confidence=confidence,
        model_id=f"deterministic+{type(llm_client).__name__}" if llm_client else "deterministic",
        invoked_agents=invoked,
    ))

    state.screening = output
    return output
