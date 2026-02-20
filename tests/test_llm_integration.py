"""Integration tests that use a real LLM via LiteLLM.

These tests are skipped unless OPENAI_API_KEY or ANTHROPIC_API_KEY is set.
They verify that the system works end-to-end with actual LLM agents.

Run with:
    OPENAI_API_KEY=sk-... pytest tests/test_llm_integration.py -v
"""

import os

import pytest

from hiring_agents.agent import run_screening
from hiring_agents.fixtures import senior_python_engineer_role, strong_match_application
from hiring_agents.llm import LiteLLMClient
from hiring_agents.state import TransactionState

has_openai = bool(os.environ.get("OPENAI_API_KEY"))
has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
skip_no_api_key = pytest.mark.skipif(
    not (has_openai or has_anthropic),
    reason="No LLM API key set (need OPENAI_API_KEY or ANTHROPIC_API_KEY)",
)

MODEL = "gpt-4o-mini" if has_openai else "claude-haiku-4-5-20251001"


@skip_no_api_key
class TestLLMIntegration:
    def test_summariser_produces_real_summary(self):
        client = LiteLLMClient(model=MODEL)
        state = TransactionState(
            application=strong_match_application(),
            role_requirements=senior_python_engineer_role(),
        )
        result = run_screening(state, llm_client=client)

        # The LLM should produce a real summary, not the simulated one
        assert "word_count" not in (state.summarisation.summary or "").lower()
        assert len(state.summarisation.summary) > 20
        assert len(state.summarisation.key_points) >= 1

    def test_reasoning_is_llm_generated(self):
        client = LiteLLMClient(model=MODEL)
        state = TransactionState(
            application=strong_match_application(),
            role_requirements=senior_python_engineer_role(),
        )
        result = run_screening(state, llm_client=client)

        # LLM reasoning should be more than the deterministic template
        assert "Evaluated" not in result.reasoning or len(result.reasoning) > 200
        # But the deterministic decision should be unchanged
        assert result.match == "strong-match"
        assert result.recommended_action == "advance"
        assert result.confidence >= 0.8

    def test_deterministic_decisions_unchanged_with_llm(self):
        """The key alignment property: LLM involvement in synthesis
        does not change the deterministic evaluation outcomes."""
        client = LiteLLMClient(model=MODEL)
        state = TransactionState(
            application=strong_match_application(),
            role_requirements=senior_python_engineer_role(),
        )
        result = run_screening(state, llm_client=client)

        # All criteria should still pass for the strong match candidate
        for c in result.criteria_evaluated:
            assert c.status == "pass"

        # Match, action, priority, confidence are deterministic
        assert result.match == "strong-match"
        assert result.recommended_action == "advance"
        assert result.priority == "urgent"

    def test_audit_trail_records_model(self):
        client = LiteLLMClient(model=MODEL)
        state = TransactionState(
            application=strong_match_application(),
            role_requirements=senior_python_engineer_role(),
        )
        run_screening(state, llm_client=client)

        # Summariser should record the real model
        summariser_record = state.decision_records[1]
        assert summariser_record.model_id == "LiteLLMClient"

        # Screening should record deterministic + real model
        screening_record = state.decision_records[2]
        assert "LiteLLMClient" in screening_record.model_id
