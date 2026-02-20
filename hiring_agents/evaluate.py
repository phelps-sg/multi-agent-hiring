"""Deterministic criterion evaluation for the Screening Agent.

Each criterion from RoleRequirements is evaluated against the
CandidateApplication data. The evaluation is deterministic — no LLM
involved. The agent cannot invent criteria beyond what is configured.

This is a key alignment mechanism: the principal (hiring manager)
defines the rules, and the agent evaluates against them transparently.
"""

from __future__ import annotations

from hiring_agents.models import (
    CandidateApplication,
    ClassificationOutput,
    CriterionResult,
    RoleRequirements,
)


def evaluate_required_skills(
    app: CandidateApplication,
    reqs: RoleRequirements,
) -> list[CriterionResult]:
    results = []
    for skill in reqs.required_skills:
        skill_lower = skill.lower()
        claimed = [s.lower() for s in app.claimed_skills]
        if skill_lower in claimed:
            results.append(CriterionResult(
                criterion=f"required_skill:{skill}",
                status="pass",
                detail=f"Candidate claims {skill}",
            ))
        else:
            results.append(CriterionResult(
                criterion=f"required_skill:{skill}",
                status="fail",
                detail=f"Candidate does not claim {skill}",
            ))
    return results


def evaluate_experience(
    app: CandidateApplication,
    reqs: RoleRequirements,
) -> list[CriterionResult]:
    results = []
    if reqs.min_years_experience is not None:
        if app.claimed_years_experience is None:
            results.append(CriterionResult(
                criterion="min_years_experience",
                status="unknown",
                detail=f"Minimum {reqs.min_years_experience} years required; candidate did not specify",
            ))
        elif app.claimed_years_experience >= reqs.min_years_experience:
            results.append(CriterionResult(
                criterion="min_years_experience",
                status="pass",
                detail=f"Candidate claims {app.claimed_years_experience} years (minimum {reqs.min_years_experience})",
            ))
        else:
            results.append(CriterionResult(
                criterion="min_years_experience",
                status="fail",
                detail=f"Candidate claims {app.claimed_years_experience} years (minimum {reqs.min_years_experience})",
            ))

    if reqs.max_years_experience is not None:
        if app.claimed_years_experience is None:
            results.append(CriterionResult(
                criterion="max_years_experience",
                status="unknown",
                detail=f"Maximum {reqs.max_years_experience} years; candidate did not specify",
            ))
        elif app.claimed_years_experience <= reqs.max_years_experience:
            results.append(CriterionResult(
                criterion="max_years_experience",
                status="pass",
                detail=f"Candidate claims {app.claimed_years_experience} years (maximum {reqs.max_years_experience})",
            ))
        else:
            results.append(CriterionResult(
                criterion="max_years_experience",
                status="fail",
                detail=f"Candidate claims {app.claimed_years_experience} years (maximum {reqs.max_years_experience})",
            ))
    return results


def evaluate_education(
    app: CandidateApplication,
    reqs: RoleRequirements,
) -> list[CriterionResult]:
    results = []
    for edu in reqs.required_education:
        edu_lower = edu.lower()
        claimed = [e.lower() for e in app.claimed_education]
        if any(edu_lower in c for c in claimed):
            results.append(CriterionResult(
                criterion=f"required_education:{edu}",
                status="pass",
                detail=f"Candidate claims {edu}",
            ))
        else:
            results.append(CriterionResult(
                criterion=f"required_education:{edu}",
                status="unknown",
                detail=f"Cannot confirm {edu} from claimed education: {app.claimed_education}",
            ))
    return results


def evaluate_location(
    app: CandidateApplication,
    reqs: RoleRequirements,
) -> list[CriterionResult]:
    if not reqs.location_requirements:
        return []
    if app.location is None:
        return [CriterionResult(
            criterion="location",
            status="unknown",
            detail=f"Required: {reqs.location_requirements}; candidate did not specify",
        )]
    loc_lower = app.location.lower()
    if any(req.lower() in loc_lower for req in reqs.location_requirements):
        return [CriterionResult(
            criterion="location",
            status="pass",
            detail=f"Candidate location '{app.location}' matches requirements",
        )]
    return [CriterionResult(
        criterion="location",
        status="fail",
        detail=f"Candidate location '{app.location}' not in {reqs.location_requirements}",
    )]


def evaluate_salary(
    app: CandidateApplication,
    reqs: RoleRequirements,
) -> list[CriterionResult]:
    results = []
    if reqs.salary_band_min is not None or reqs.salary_band_max is not None:
        if app.salary_expectation is None:
            results.append(CriterionResult(
                criterion="salary",
                status="unknown",
                detail="Candidate did not specify salary expectation",
            ))
        else:
            band = f"{reqs.salary_band_min or '?'}–{reqs.salary_band_max or '?'}"
            if reqs.salary_band_max is not None and app.salary_expectation > reqs.salary_band_max:
                results.append(CriterionResult(
                    criterion="salary",
                    status="fail",
                    detail=f"Candidate expects {app.salary_expectation}; band is {band}",
                ))
            elif reqs.salary_band_min is not None and app.salary_expectation < reqs.salary_band_min:
                results.append(CriterionResult(
                    criterion="salary",
                    status="pass",
                    detail=f"Candidate expects {app.salary_expectation} (below band {band})",
                ))
            else:
                results.append(CriterionResult(
                    criterion="salary",
                    status="pass",
                    detail=f"Candidate expects {app.salary_expectation}; within band {band}",
                ))
    return results


def evaluate_exclusion_triggers(
    app: CandidateApplication,
    reqs: RoleRequirements,
) -> list[CriterionResult]:
    """Check for hard-reject triggers.

    In practice, these would be checked against extracted CV content.
    Here we check against document content as a simple demonstration.
    """
    results = []
    all_content = " ".join(doc.content.lower() for doc in app.documents)
    for trigger in reqs.exclusion_triggers:
        if trigger.lower() in all_content:
            results.append(CriterionResult(
                criterion=f"exclusion_trigger:{trigger}",
                status="fail",
                detail=f"Exclusion trigger '{trigger}' found in application documents",
            ))
        else:
            results.append(CriterionResult(
                criterion=f"exclusion_trigger:{trigger}",
                status="pass",
                detail=f"Exclusion trigger '{trigger}' not found",
            ))
    return results


def evaluate_required_documents(
    classification: ClassificationOutput | None,
    reqs: RoleRequirements,
) -> list[CriterionResult]:
    if not reqs.required_documents:
        return []
    if classification is None:
        return [CriterionResult(
            criterion="required_documents",
            status="unknown",
            detail="Document classification not yet available",
        )]
    results = []
    for doc_type in reqs.required_documents:
        if doc_type in classification.documents_present:
            results.append(CriterionResult(
                criterion=f"required_document:{doc_type}",
                status="pass",
                detail=f"Document type '{doc_type}' present",
            ))
        else:
            results.append(CriterionResult(
                criterion=f"required_document:{doc_type}",
                status="fail",
                detail=f"Required document type '{doc_type}' missing",
            ))
    return results


def evaluate_all_criteria(
    app: CandidateApplication,
    reqs: RoleRequirements,
    classification: ClassificationOutput | None = None,
) -> list[CriterionResult]:
    """Evaluate all configured criteria. Returns only criteria that exist
    in the RoleRequirements — the agent cannot invent additional criteria.
    """
    results: list[CriterionResult] = []
    results.extend(evaluate_required_skills(app, reqs))
    results.extend(evaluate_experience(app, reqs))
    results.extend(evaluate_education(app, reqs))
    results.extend(evaluate_location(app, reqs))
    results.extend(evaluate_salary(app, reqs))
    results.extend(evaluate_exclusion_triggers(app, reqs))
    results.extend(evaluate_required_documents(classification, reqs))
    return results
