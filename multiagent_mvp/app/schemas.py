from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


AgentQuestionCategory = Literal[
    "scope_clarification",
    "business_rule_clarification",
    "api_contract_clarification",
    "ux_behavior_clarification",
    "risk_confirmation",
]


class AgentQuestion(BaseModel):
    from_role: str
    to_role: str
    category: AgentQuestionCategory
    question: str
    reason: str
    blocking: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentAnswer(BaseModel):
    from_role: str
    to_role: str
    answer: str
    decision: str
    constraints: List[str] = Field(default_factory=list)
    resolved: bool


class AgentDialogueTurn(BaseModel):
    question: AgentQuestion
    answer: AgentAnswer


class ArchitectureReviewerReport(BaseModel):
    summary: str = Field(description="Resumen ejecutivo de la revisión arquitectónica")
    strengths: List[str] = Field(default_factory=list, description="Fortalezas detectadas")
    issues: List[str] = Field(default_factory=list, description="Problemas o inconsistencias detectadas")
    mvp_simplifications: List[str] = Field(
        default_factory=list,
        description="Simplificaciones recomendadas para un MVP",
    )
    security_observations: List[str] = Field(
        default_factory=list,
        description="Observaciones de seguridad, privacidad y compliance",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recomendaciones accionables",
    )
    approval_status: str = Field(
        description="Estado final de revisión. Valores sugeridos: approved, approved_with_changes, rejected"
    )
    
class ProjectCreateRequest(BaseModel):
    title: str = Field(..., min_length=3)
    goal: str = Field(..., min_length=10)
    constraints: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)


class ProductBrief(BaseModel):
    product_summary: str
    business_goals: List[str]
    personas: List[str]
    epics: List[str]
    risks: List[str]


class RequirementItem(BaseModel):
    id: str
    title: str
    description: str
    acceptance_criteria: List[str]
    priority: Literal["high", "medium", "low"]


class RequirementsSpec(BaseModel):
    scope: str
    business_rules: List[str]
    user_stories: List[RequirementItem]
    open_questions: List[str]


class BackendEndpoint(BaseModel):
    method: str
    path: str
    purpose: str


class BackendSpec(BaseModel):
    architecture: str
    domain_entities: List[str]
    api_endpoints: List[BackendEndpoint]
    data_model: List[str]
    technical_risks: List[str]


class FrontendScreen(BaseModel):
    name: str
    purpose: str
    main_components: List[str]


class FrontendSpec(BaseModel):
    app_structure: str
    screens: List[FrontendScreen]
    state_management: str
    integration_points: List[str]
    ux_risks: List[str]


class TestCase(BaseModel):
    id: str
    title: str
    objective: str
    preconditions: List[str]
    expected_result: str


class TestPlan(BaseModel):
    strategy: str
    coverage: List[str]
    test_cases: List[TestCase]
    release_gates: List[str]


class ReleaseBundle(BaseModel):
    branch_strategy: str
    commit_plan: List[str]
    pull_request_title: str
    release_notes: List[str]
    checklist: List[str]


class ProjectArtifacts(BaseModel):
    product_brief: ProductBrief | None = None
    requirements_spec: RequirementsSpec | None = None
    backend_spec: BackendSpec | None = None
    frontend_spec: FrontendSpec | None = None
    architecture_review: ArchitectureReviewerReport | None = None
    test_plan: TestPlan | None = None
    release_bundle: ReleaseBundle | None = None


class ProjectState(BaseModel):
    project_id: str
    title: str
    goal: str
    current_phase: str
    constraints: List[str]
    context: Dict[str, Any]
    artifacts: ProjectArtifacts = Field(default_factory=ProjectArtifacts)
    audit_log: List[str] = Field(default_factory=list)
    agent_dialogue: List[AgentDialogueTurn] = Field(default_factory=list)


class ProjectResponse(BaseModel):
    project_id: str
    status: Literal["completed", "blocked"]
    state: ProjectState
