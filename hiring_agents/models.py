"""Domain models for the multi-agent hiring system.

Structured types for role requirements (the principal's configuration),
candidate applications (adversarial input), and agent outputs at each tier.
"""

from __future__ import annotations

import typing as t
from datetime import datetime

import msgspec


# --- Principal's Configuration (Role Requirements) ---


class RoleRequirements(msgspec.Struct, kw_only=True):
    """Structured rules defining what the hiring manager wants.

    Analogous to AppetiteConfig in insurance. The screening agent evaluates
    each criterion and reports pass/fail/unknown — it cannot invent criteria
    beyond what is configured here.
    """

    role_title: str
    department: str
    required_skills: list[str]
    preferred_skills: list[str] = []
    min_years_experience: int | None = None
    max_years_experience: int | None = None
    required_education: list[str] = []  # e.g. ["BSc Computer Science", "MSc"]
    location_requirements: list[str] = []  # e.g. ["London", "Remote UK"]
    salary_band_min: float | None = None
    salary_band_max: float | None = None
    exclusion_triggers: list[str] = []  # hard-reject triggers, e.g. ["no right to work"]
    required_documents: list[str] = []  # e.g. ["cv", "cover_letter"]


# --- Candidate Application (Adversarial Input) ---


class Document(msgspec.Struct, kw_only=True):
    """A document submitted as part of an application."""

    document_id: str
    document_type: str  # e.g. "cv", "cover_letter", "portfolio", "references"
    content: str
    metadata: dict[str, t.Any] = {}


class CandidateApplication(msgspec.Struct, kw_only=True):
    """A candidate's application — treated as adversarial input to be verified."""

    application_id: str
    candidate_name: str
    email: str
    documents: list[Document]
    claimed_skills: list[str] = []
    claimed_years_experience: int | None = None
    claimed_education: list[str] = []
    location: str | None = None
    salary_expectation: float | None = None


# --- Agent Outputs ---


class ClassificationOutput(msgspec.Struct, kw_only=True):
    """Output of the Document Classification Agent."""

    application_id: str
    document_classifications: dict[str, str]  # doc_id -> classified type
    documents_present: list[str]  # list of document types found
    documents_missing: list[str]  # required docs not found


class SummarisationOutput(msgspec.Struct, kw_only=True):
    """Output of the Summarisation Agent."""

    application_id: str
    summary: str
    focus: str  # what aspect was emphasised
    key_points: list[str]


CriterionStatus = t.Literal["pass", "fail", "unknown"]


class CriterionResult(msgspec.Struct, kw_only=True):
    """Result of evaluating a single screening criterion."""

    criterion: str
    status: CriterionStatus
    detail: str


class RiskFlag(msgspec.Struct, kw_only=True):
    """A flag raised during screening."""

    flag: str
    severity: t.Literal["high", "medium", "low"]
    detail: str


class ScreeningOutput(msgspec.Struct, kw_only=True):
    """Output of the Screening Agent (Tier 1 — analogous to Triage).

    The screening agent evaluates each criterion from RoleRequirements
    deterministically. It cannot invent criteria beyond what's configured.
    """

    application_id: str
    match: t.Literal["strong-match", "possible-match", "no-match"]
    confidence: float
    priority: t.Literal["urgent", "standard", "low"]
    recommended_action: t.Literal["advance", "refer-to-hiring-manager", "reject", "request-info"]
    missing_information: list[str]
    flags: list[RiskFlag]
    reasoning: str
    criteria_evaluated: list[CriterionResult]


class CandidateProfileOutput(msgspec.Struct, kw_only=True):
    """Output of the Candidate Profile Agent (Tier 1 — analogous to Risk Profile)."""

    application_id: str
    skills_assessment: dict[str, str]  # skill -> assessment
    experience_assessment: str
    education_assessment: str
    strengths: list[str]
    gaps: list[str]
    inconsistencies: list[str]
    completeness_score: float
    reasoning_trace: list[str]


class InterviewRecommendation(msgspec.Struct, kw_only=True):
    """Output of the Interview Recommendation Agent (Tier 1 — analogous to Quote)."""

    application_id: str
    recommend_interview: bool
    interview_format: t.Literal["phone-screen", "technical", "panel", "skip"] | None
    areas_to_probe: list[str]
    confidence: float
    reasoning: str
    requires_senior_review: bool
    senior_review_triggers: list[str]


# --- Observable State & Audit Trail ---


class ReasoningStep(msgspec.Struct, kw_only=True):
    """A single step in an agent's reasoning trace."""

    step: str
    detail: str


class AgentDecisionRecord(msgspec.Struct, kw_only=True):
    """Audit record for every agent decision.

    Every agent invocation produces one of these. They are stored in
    the shared transaction state, making the entire decision chain
    observable and auditable.
    """

    agent_id: str
    timestamp: datetime
    inputs_summary: dict[str, t.Any]
    reasoning_steps: list[ReasoningStep]
    output: t.Any
    confidence: float
    model_id: str
    human_override: dict[str, t.Any] | None = None
    invoked_agents: list[str] = []
