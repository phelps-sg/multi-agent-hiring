"""Observable shared state between agents.

All agents interact through this state object — no direct agent-to-agent
communication. This makes every interaction auditable and prevents
emergent coordination failures.

In production, this would be transaction fields in S3 + event queue in SQS.
Here it's an in-memory object for demonstration.
"""

from __future__ import annotations

from hiring_agents.models import (
    AgentDecisionRecord,
    CandidateApplication,
    CandidateProfileOutput,
    ClassificationOutput,
    InterviewRecommendation,
    RoleRequirements,
    ScreeningOutput,
    SummarisationOutput,
)


class TransactionState:
    """Observable shared state for a single application's processing.

    Agents read inputs from here and write outputs back.
    The decision_records list provides a complete audit trail.
    """

    def __init__(
        self,
        application: CandidateApplication,
        role_requirements: RoleRequirements,
    ) -> None:
        self.application = application
        self.role_requirements = role_requirements

        # Agent outputs — populated as each agent runs
        self.classification: ClassificationOutput | None = None
        self.summarisation: SummarisationOutput | None = None
        self.screening: ScreeningOutput | None = None
        self.candidate_profile: CandidateProfileOutput | None = None
        self.interview_recommendation: InterviewRecommendation | None = None

        # Audit trail
        self.decision_records: list[AgentDecisionRecord] = []

    def record_decision(self, record: AgentDecisionRecord) -> None:
        """Append a decision record to the audit trail."""
        self.decision_records.append(record)
