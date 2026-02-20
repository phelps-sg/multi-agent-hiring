"""Test fixtures: sample role requirements and candidate applications.

These demonstrate the different screening outcomes:
- Strong match (advance)
- No match — missing required skills (reject)
- Possible match — missing documents (request-info)
- No match — exclusion trigger (reject)
- No match — salary above band (reject via flag)
"""

from hiring_agents.models import CandidateApplication, Document, RoleRequirements


def senior_python_engineer_role() -> RoleRequirements:
    return RoleRequirements(
        role_title="Senior Python Engineer",
        department="Engineering",
        required_skills=["python", "sql", "git"],
        preferred_skills=["kubernetes", "aws", "terraform"],
        min_years_experience=5,
        max_years_experience=15,
        required_education=["BSc"],
        location_requirements=["London", "Remote UK"],
        salary_band_min=80000,
        salary_band_max=120000,
        exclusion_triggers=["no right to work in UK"],
        required_documents=["cv", "cover_letter"],
    )


def strong_match_application() -> CandidateApplication:
    """Candidate who matches all criteria — should be advanced."""
    return CandidateApplication(
        application_id="app-001",
        candidate_name="Alice Chen",
        email="alice@example.com",
        documents=[
            Document(
                document_id="doc-001",
                document_type="cv",
                content=(
                    "Curriculum Vitae — Alice Chen\n\n"
                    "Senior Software Engineer with 8 years of experience in Python, "
                    "SQL, and cloud infrastructure. BSc Computer Science from UCL. "
                    "Currently leading a team of 5 engineers building data pipelines "
                    "with Kubernetes on AWS. Proficient in Git, Terraform, and CI/CD."
                ),
            ),
            Document(
                document_id="doc-002",
                document_type="cover_letter",
                content=(
                    "Dear Hiring Manager,\n\n"
                    "I am writing to apply for the Senior Python Engineer role. "
                    "With 8 years of experience in backend systems and a strong "
                    "background in Python and SQL, I believe I would be a strong "
                    "fit for your team. I am based in London and available to start "
                    "within one month."
                ),
            ),
        ],
        claimed_skills=["python", "sql", "git", "kubernetes", "aws", "terraform"],
        claimed_years_experience=8,
        claimed_education=["BSc Computer Science"],
        location="London",
        salary_expectation=100000,
    )


def missing_skills_application() -> CandidateApplication:
    """Candidate missing required skills — should be rejected."""
    return CandidateApplication(
        application_id="app-002",
        candidate_name="Bob Smith",
        email="bob@example.com",
        documents=[
            Document(
                document_id="doc-003",
                document_type="cv",
                content=(
                    "CV — Bob Smith\n\n"
                    "Junior Frontend Developer with 2 years of experience. "
                    "Skilled in JavaScript, React, and CSS. Looking to transition "
                    "into backend development."
                ),
            ),
            Document(
                document_id="doc-004",
                document_type="cover_letter",
                content=(
                    "Dear Hiring Manager,\n\n"
                    "I am writing to apply for the Senior Python Engineer position. "
                    "While my background is in frontend development, I am eager to "
                    "learn Python and backend technologies."
                ),
            ),
        ],
        claimed_skills=["javascript", "react", "css"],
        claimed_years_experience=2,
        claimed_education=["BSc Web Design"],
        location="London",
        salary_expectation=85000,
    )


def missing_documents_application() -> CandidateApplication:
    """Candidate with missing cover letter — should request info."""
    return CandidateApplication(
        application_id="app-003",
        candidate_name="Carol Davis",
        email="carol@example.com",
        documents=[
            Document(
                document_id="doc-005",
                document_type="cv",
                content=(
                    "Curriculum Vitae — Carol Davis\n\n"
                    "Software Engineer with 6 years of Python experience. "
                    "Strong SQL skills, extensive Git usage, BSc Mathematics. "
                    "Based in Manchester, open to remote UK positions."
                ),
            ),
            # No cover letter submitted
        ],
        claimed_skills=["python", "sql", "git"],
        claimed_years_experience=6,
        claimed_education=["BSc Mathematics"],
        location="Manchester",
        salary_expectation=95000,
    )


def exclusion_trigger_application() -> CandidateApplication:
    """Candidate who triggers an exclusion — should be rejected immediately."""
    return CandidateApplication(
        application_id="app-004",
        candidate_name="Dave Wilson",
        email="dave@example.com",
        documents=[
            Document(
                document_id="doc-006",
                document_type="cv",
                content=(
                    "CV — Dave Wilson\n\n"
                    "Experienced Python developer, 7 years. SQL, Git, AWS. "
                    "BSc Computer Science. Note: I currently have no right to work "
                    "in UK and would require visa sponsorship."
                ),
            ),
            Document(
                document_id="doc-007",
                document_type="cover_letter",
                content=(
                    "Dear Hiring Manager,\n\n"
                    "I am applying for the Senior Python Engineer role. "
                    "I have extensive experience in Python and cloud technologies."
                ),
            ),
        ],
        claimed_skills=["python", "sql", "git", "aws"],
        claimed_years_experience=7,
        claimed_education=["BSc Computer Science"],
        location="Remote",
        salary_expectation=110000,
    )


def salary_above_band_application() -> CandidateApplication:
    """Candidate expecting salary well above band — flagged, rejected."""
    return CandidateApplication(
        application_id="app-005",
        candidate_name="Eve Martinez",
        email="eve@example.com",
        documents=[
            Document(
                document_id="doc-008",
                document_type="cv",
                content=(
                    "CV — Eve Martinez\n\n"
                    "Staff Engineer with 12 years of Python experience. "
                    "Led teams at major tech companies. BSc and MSc in CS. "
                    "Expert in distributed systems, Kubernetes, Terraform."
                ),
            ),
            Document(
                document_id="doc-009",
                document_type="cover_letter",
                content=(
                    "Dear Hiring Manager,\n\n"
                    "I am writing to apply for the Senior Python Engineer role. "
                    "I bring extensive experience in Python and infrastructure."
                ),
            ),
        ],
        claimed_skills=["python", "sql", "git", "kubernetes", "terraform", "aws"],
        claimed_years_experience=12,
        claimed_education=["BSc Computer Science", "MSc Computer Science"],
        location="London",
        salary_expectation=180000,  # Well above 120k max
    )
