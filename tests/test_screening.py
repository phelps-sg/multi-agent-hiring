"""Tests for the multi-agent screening pipeline.

Tests cover:
- End-to-end scenarios (strong match, no match, borderline, exclusion, salary)
- Data flow through shared state
- Audit trail completeness
- Criterion evaluation correctness
- Alignment: agent cannot invent criteria beyond RoleRequirements
"""

from hiring_agents.agent import run_screening
from hiring_agents.evaluate import evaluate_all_criteria
from hiring_agents.fixtures import (
    exclusion_trigger_application,
    missing_documents_application,
    missing_skills_application,
    salary_above_band_application,
    senior_python_engineer_role,
    strong_match_application,
)
from hiring_agents.models import CandidateApplication, Document, RoleRequirements
from hiring_agents.state import TransactionState


# --- End-to-End Scenarios ---


class TestStrongMatch:
    def test_strong_match_advances(self):
        state = TransactionState(
            application=strong_match_application(),
            role_requirements=senior_python_engineer_role(),
        )
        result = run_screening(state)

        assert result.match == "strong-match"
        assert result.recommended_action == "advance"
        assert result.confidence >= 0.8

    def test_strong_match_priority_urgent(self):
        state = TransactionState(
            application=strong_match_application(),
            role_requirements=senior_python_engineer_role(),
        )
        result = run_screening(state)
        assert result.priority == "urgent"


class TestMissingSkills:
    def test_missing_skills_rejected(self):
        state = TransactionState(
            application=missing_skills_application(),
            role_requirements=senior_python_engineer_role(),
        )
        result = run_screening(state)

        assert result.match == "no-match"
        assert result.recommended_action == "reject"

    def test_missing_skills_identified(self):
        state = TransactionState(
            application=missing_skills_application(),
            role_requirements=senior_python_engineer_role(),
        )
        result = run_screening(state)

        failed_criteria = [c for c in result.criteria_evaluated if c.status == "fail"]
        failed_names = [c.criterion for c in failed_criteria]
        assert "required_skill:python" in failed_names
        assert "required_skill:sql" in failed_names
        assert "required_skill:git" in failed_names


class TestMissingDocuments:
    def test_missing_docs_requests_info(self):
        state = TransactionState(
            application=missing_documents_application(),
            role_requirements=senior_python_engineer_role(),
        )
        result = run_screening(state)

        # Missing cover letter should trigger request-info or refer
        assert "document:cover_letter" in result.missing_information or any(
            c.criterion == "required_document:cover_letter" and c.status == "fail"
            for c in result.criteria_evaluated
        )


class TestExclusionTrigger:
    def test_exclusion_trigger_rejects(self):
        state = TransactionState(
            application=exclusion_trigger_application(),
            role_requirements=senior_python_engineer_role(),
        )
        result = run_screening(state)

        assert result.match == "no-match"
        assert result.recommended_action == "reject"
        assert result.confidence >= 0.9

    def test_exclusion_trigger_identified(self):
        state = TransactionState(
            application=exclusion_trigger_application(),
            role_requirements=senior_python_engineer_role(),
        )
        result = run_screening(state)

        exclusion_criteria = [
            c for c in result.criteria_evaluated
            if c.criterion.startswith("exclusion_trigger:") and c.status == "fail"
        ]
        assert len(exclusion_criteria) >= 1


class TestSalaryAboveBand:
    def test_salary_above_band_flagged(self):
        state = TransactionState(
            application=salary_above_band_application(),
            role_requirements=senior_python_engineer_role(),
        )
        result = run_screening(state)

        assert result.match == "no-match"
        assert result.recommended_action == "reject"
        salary_flags = [f for f in result.flags if f.flag == "salary_significantly_above_band"]
        assert len(salary_flags) == 1
        assert salary_flags[0].severity == "high"


# --- Data Flow Through Shared State ---


class TestSharedState:
    def test_classification_populated(self):
        state = TransactionState(
            application=strong_match_application(),
            role_requirements=senior_python_engineer_role(),
        )
        run_screening(state)

        assert state.classification is not None
        assert len(state.classification.document_classifications) == 2
        assert "cv" in state.classification.documents_present
        assert "cover_letter" in state.classification.documents_present

    def test_summarisation_populated(self):
        state = TransactionState(
            application=strong_match_application(),
            role_requirements=senior_python_engineer_role(),
        )
        run_screening(state)

        assert state.summarisation is not None
        assert state.summarisation.focus == "screening"

    def test_screening_populated(self):
        state = TransactionState(
            application=strong_match_application(),
            role_requirements=senior_python_engineer_role(),
        )
        run_screening(state)

        assert state.screening is not None
        assert state.screening.application_id == "app-001"


# --- Audit Trail ---


class TestAuditTrail:
    def test_three_decision_records(self):
        """Each run should produce exactly 3 decision records:
        classifier, summariser, screening agent."""
        state = TransactionState(
            application=strong_match_application(),
            role_requirements=senior_python_engineer_role(),
        )
        run_screening(state)

        assert len(state.decision_records) == 3
        agent_ids = [r.agent_id for r in state.decision_records]
        assert agent_ids == ["classifier:001", "summariser:001", "screening:001"]

    def test_screening_record_tracks_invoked_agents(self):
        state = TransactionState(
            application=strong_match_application(),
            role_requirements=senior_python_engineer_role(),
        )
        run_screening(state)

        screening_record = state.decision_records[2]
        assert screening_record.agent_id == "screening:001"
        assert "classifier:001" in screening_record.invoked_agents
        assert "summariser:001" in screening_record.invoked_agents

    def test_all_records_have_timestamps(self):
        state = TransactionState(
            application=strong_match_application(),
            role_requirements=senior_python_engineer_role(),
        )
        run_screening(state)

        for record in state.decision_records:
            assert record.timestamp is not None


# --- Criterion Evaluation ---


class TestCriterionEvaluation:
    def test_skill_pass(self):
        app = strong_match_application()
        reqs = senior_python_engineer_role()
        criteria = evaluate_all_criteria(app, reqs)

        python_criteria = [c for c in criteria if c.criterion == "required_skill:python"]
        assert len(python_criteria) == 1
        assert python_criteria[0].status == "pass"

    def test_skill_fail(self):
        app = missing_skills_application()
        reqs = senior_python_engineer_role()
        criteria = evaluate_all_criteria(app, reqs)

        python_criteria = [c for c in criteria if c.criterion == "required_skill:python"]
        assert len(python_criteria) == 1
        assert python_criteria[0].status == "fail"

    def test_experience_unknown_when_not_specified(self):
        app = CandidateApplication(
            application_id="test",
            candidate_name="Test",
            email="test@test.com",
            documents=[],
            claimed_years_experience=None,
        )
        reqs = RoleRequirements(
            role_title="Test",
            department="Test",
            required_skills=[],
            min_years_experience=3,
        )
        criteria = evaluate_all_criteria(app, reqs)

        exp_criteria = [c for c in criteria if c.criterion == "min_years_experience"]
        assert len(exp_criteria) == 1
        assert exp_criteria[0].status == "unknown"

    def test_salary_fail_above_band(self):
        app = salary_above_band_application()
        reqs = senior_python_engineer_role()
        criteria = evaluate_all_criteria(app, reqs)

        salary_criteria = [c for c in criteria if c.criterion == "salary"]
        assert len(salary_criteria) == 1
        assert salary_criteria[0].status == "fail"

    def test_location_fail_outside_scope(self):
        app = CandidateApplication(
            application_id="test",
            candidate_name="Test",
            email="test@test.com",
            documents=[],
            location="New York",
        )
        reqs = RoleRequirements(
            role_title="Test",
            department="Test",
            required_skills=[],
            location_requirements=["London", "Remote UK"],
        )
        criteria = evaluate_all_criteria(app, reqs)

        loc_criteria = [c for c in criteria if c.criterion == "location"]
        assert len(loc_criteria) == 1
        assert loc_criteria[0].status == "fail"


# --- Alignment Tests ---


class TestAlignment:
    def test_only_configured_criteria_evaluated(self):
        """The agent must only evaluate criteria that exist in RoleRequirements.
        It cannot invent additional criteria."""
        reqs = RoleRequirements(
            role_title="Minimal Role",
            department="Test",
            required_skills=["python"],
            # Everything else is empty/None — no other criteria configured
        )
        app = CandidateApplication(
            application_id="test",
            candidate_name="Test",
            email="test@test.com",
            documents=[],
            claimed_skills=["python"],
        )
        criteria = evaluate_all_criteria(app, reqs)

        # Should only have the one required skill criterion
        criterion_types = [c.criterion for c in criteria]
        assert criterion_types == ["required_skill:python"]

    def test_empty_requirements_produces_no_criteria(self):
        """If no criteria are configured, none should be evaluated."""
        reqs = RoleRequirements(
            role_title="Open Role",
            department="Test",
            required_skills=[],
        )
        app = CandidateApplication(
            application_id="test",
            candidate_name="Test",
            email="test@test.com",
            documents=[],
        )
        criteria = evaluate_all_criteria(app, reqs)
        assert criteria == []

    def test_recommend_only_no_binding_decisions(self):
        """Screening output should always be a recommendation,
        never a binding decision."""
        state = TransactionState(
            application=strong_match_application(),
            role_requirements=senior_python_engineer_role(),
        )
        result = run_screening(state)

        # The recommended_action field is always a recommendation
        assert result.recommended_action in [
            "advance", "refer-to-hiring-manager", "reject", "request-info",
        ]
        # There is no "hire" or "offer" action — the agent cannot make binding decisions
