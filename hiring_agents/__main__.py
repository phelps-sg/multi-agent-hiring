"""Run the multi-agent screening pipeline.

Usage:
    # With simulated LLM (no API key needed):
    python -m hiring_agents

    # With a real LLM via LiteLLM:
    python -m hiring_agents --model gpt-4o-mini

    # With any LiteLLM-supported model:
    python -m hiring_agents --model claude-sonnet-4-20250514
"""

from __future__ import annotations

import argparse
import json

from hiring_agents.agent import run_screening
from hiring_agents.fixtures import (
    exclusion_trigger_application,
    missing_documents_application,
    missing_skills_application,
    salary_above_band_application,
    senior_python_engineer_role,
    strong_match_application,
)
from hiring_agents.state import TransactionState
from hiring_agents.sub_agents.summariser import LLMClient


def _format_output(result, state: TransactionState) -> str:
    lines = []
    lines.append(f"  Candidate: {state.application.candidate_name}")
    lines.append(f"  Match:     {result.match}")
    lines.append(f"  Action:    {result.recommended_action}")
    lines.append(f"  Confidence: {result.confidence:.0%}")
    lines.append(f"  Priority:  {result.priority}")
    if result.flags:
        lines.append(f"  Flags:     {', '.join(f.flag for f in result.flags)}")
    if result.missing_information:
        lines.append(f"  Missing:   {', '.join(result.missing_information)}")
    lines.append(f"  Reasoning: {result.reasoning}")
    lines.append(f"  Criteria:  {len(result.criteria_evaluated)} evaluated")
    for c in result.criteria_evaluated:
        lines.append(f"    {c.status:>7s}  {c.criterion} — {c.detail}")
    lines.append(f"  Audit trail: {len(state.decision_records)} decision records")
    for rec in state.decision_records:
        lines.append(f"    [{rec.agent_id}] confidence={rec.confidence:.0%} model={rec.model_id}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the multi-agent screening pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LiteLLM model name (e.g. gpt-4o-mini, claude-sonnet-4-20250514). "
        "If not specified, uses simulated LLM.",
    )
    args = parser.parse_args()

    llm_client: LLMClient | None = None
    if args.model:
        from hiring_agents.llm import LiteLLMClient

        llm_client = LiteLLMClient(model=args.model)
        print(f"Using LLM: {args.model}")
    else:
        print("Using simulated LLM (pass --model to use a real one)")

    print()

    role = senior_python_engineer_role()
    print(f"Role: {role.role_title} ({role.department})")
    print(f"Required skills: {', '.join(role.required_skills)}")
    print(f"Experience: {role.min_years_experience}-{role.max_years_experience} years")
    print(f"Salary band: {role.salary_band_min:,.0f}-{role.salary_band_max:,.0f}")
    print()

    scenarios = [
        ("Strong match", strong_match_application),
        ("Missing skills", missing_skills_application),
        ("Missing documents", missing_documents_application),
        ("Exclusion trigger", exclusion_trigger_application),
        ("Salary above band", salary_above_band_application),
    ]

    for label, make_app in scenarios:
        print(f"--- {label} ---")
        state = TransactionState(
            application=make_app(),
            role_requirements=role,
        )
        result = run_screening(state, llm_client=llm_client)
        print(_format_output(result, state))
        print()


if __name__ == "__main__":
    main()
