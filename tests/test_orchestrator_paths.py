from __future__ import annotations

import asyncio

from app.knowledge import build_default_knowledge_provider
from app.orchestrator import PipelineOrchestrator
import app.orchestrator as orchestrator_module
from app.schemas import (
    ArchitectureReviewerReport,
    BackendEndpoint,
    BackendSpec,
    FrontendScreen,
    FrontendSpec,
    ProductBrief,
    ProjectCreateRequest,
    ProjectArtifacts,
    ProjectState,
    ReleaseBundle,
    RequirementItem,
    RequirementsSpec,
    TestCase as QaTestCase,
    TestPlan as QaTestPlan,
)


def _request() -> ProjectCreateRequest:
    return ProjectCreateRequest(
        title="Proyecto demo",
        goal="Construir una solución de onboarding de clientes.",
        constraints=["Debe tener autenticación"],
        context={"priority": "alta"},
    )


def _ok_payloads():
    return {
        "product_owner": ProductBrief(
            product_summary="Resumen",
            business_goals=["Goal 1"],
            personas=["Persona 1"],
            epics=["Epic 1"],
            risks=["Risk 1"],
        ),
        "functional_analyst": RequirementsSpec(
            scope="MVP",
            business_rules=["Rule 1"],
            user_stories=[
                RequirementItem(
                    id="US-1",
                    title="Story",
                    description="Desc",
                    acceptance_criteria=["AC-1"],
                    priority="high",
                )
            ],
            open_questions=[],
        ),
        "backend_developer": BackendSpec(
            architecture="Modular",
            domain_entities=["Customer"],
            api_endpoints=[BackendEndpoint(method="GET", path="/customers", purpose="Listar")],
            data_model=["customers(id, name)"],
            technical_risks=[],
        ),
        "frontend_developer": FrontendSpec(
            app_structure="SPA",
            screens=[FrontendScreen(name="Home", purpose="Inicio", main_components=["Table"])],
            state_management="local",
            integration_points=["/customers"],
            ux_risks=[],
        ),
        "architecture_reviewer": ArchitectureReviewerReport(
            summary="OK",
            strengths=["Consistente"],
            issues=[],
            mvp_simplifications=[],
            security_observations=[],
            recommendations=[],
            approval_status="approved",
        ),
        "qa_analyst": QaTestPlan(
            strategy="Risk based",
            coverage=["Core flows"],
            test_cases=[QaTestCase(
                    id="TC-1",
                    title="Caso",
                    objective="Validar",
                    preconditions=[],
                    expected_result="OK",
                )],
            release_gates=["Gate: smoke tests pass"],
        ),
        "commit_manager": ReleaseBundle(
            branch_strategy="trunk-based",
            commit_plan=["feat: init"],
            pull_request_title="PR",
            release_notes=["note"],
            checklist=["check"],
        ),
    }


def test_pipeline_completed_path():
    orch = PipelineOrchestrator(build_default_knowledge_provider())
    payloads = _ok_payloads()

    async def fake_run_role(role, agent_builder, input_payload):
        return payloads[role]

    orch._run_role = fake_run_role  # type: ignore[method-assign]
    response = asyncio.run(orch.run_project(_request()))

    assert response.status == "completed"
    assert response.state.current_phase == "completed"


def test_pipeline_blocked_by_architecture_review():
    orch = PipelineOrchestrator(build_default_knowledge_provider())
    payloads = _ok_payloads()
    payloads["architecture_reviewer"] = payloads["architecture_reviewer"].model_copy(
        update={"approval_status": "rejected"}
    )

    async def fake_run_role(role, agent_builder, input_payload):
        return payloads[role]

    orch._run_role = fake_run_role  # type: ignore[method-assign]
    response = asyncio.run(orch.run_project(_request()))

    assert response.status == "blocked"
    assert response.state.current_phase == "blocked_architecture_review"


def test_pipeline_blocked_by_qa_gates():
    orch = PipelineOrchestrator(build_default_knowledge_provider())
    payloads = _ok_payloads()
    payloads["qa_analyst"] = payloads["qa_analyst"].model_copy(
        update={"release_gates": ["BLOCKER: regression critical fail"]}
    )

    async def fake_run_role(role, agent_builder, input_payload):
        return payloads[role]

    orch._run_role = fake_run_role  # type: ignore[method-assign]
    response = asyncio.run(orch.run_project(_request()))

    assert response.status == "blocked"
    assert response.state.current_phase == "blocked_qa_gate"


def test_pipeline_accepts_tuple_role_outputs():
    orch = PipelineOrchestrator(build_default_knowledge_provider())
    payloads = _ok_payloads()

    async def fake_run_role(role, agent_builder, input_payload):
        return payloads[role], {"meta": "ok"}

    orch._run_role = fake_run_role  # type: ignore[method-assign]
    response = asyncio.run(orch.run_project(_request()))

    assert response.status == "completed"
    assert response.state.artifacts.product_brief is not None


def test_pipeline_accepts_tuple_outputs_when_first_item_is_bool_placeholder():
    orch = PipelineOrchestrator(build_default_knowledge_provider())
    payloads = _ok_payloads()

    async def fake_run_role(role, agent_builder, input_payload):
        return False, payloads[role]

    orch._run_role = fake_run_role  # type: ignore[method-assign]
    response = asyncio.run(orch.run_project(_request()))

    assert response.status == "completed"
    assert response.state.artifacts.release_bundle is not None


def test_coerce_output_supports_model_dict_json_bytes_singleton_list():
    orch = PipelineOrchestrator(build_default_knowledge_provider())
    model = ProductBrief(
        product_summary="x",
        business_goals=["g"],
        personas=["p"],
        epics=["e"],
        risks=["r"],
    )

    as_dict = model.model_dump()
    as_json = model.model_dump_json()
    as_bytes = as_json.encode("utf-8")

    assert orch._coerce_output(ProductBrief, model).product_summary == "x"
    assert orch._coerce_output(ProductBrief, as_dict).product_summary == "x"
    assert orch._coerce_output(ProductBrief, as_json).product_summary == "x"
    assert orch._coerce_output(ProductBrief, as_bytes).product_summary == "x"
    assert orch._coerce_output(ProductBrief, [as_dict]).product_summary == "x"


def test_coerce_output_rejects_bool_with_clear_error():
    orch = PipelineOrchestrator(build_default_knowledge_provider())
    try:
        orch._coerce_output(ProductBrief, True)
    except ValueError as exc:
        assert "bool" in str(exc)
    else:
        raise AssertionError("Expected ValueError for bool output")


def test_extract_raw_output_supports_sdk_variants():
    orch = PipelineOrchestrator(build_default_knowledge_provider())
    expected = {"product_summary": "x", "business_goals": ["g"], "personas": ["p"], "epics": ["e"], "risks": ["r"]}

    class ResultA:
        final_output = expected

    class ResultB:
        final_output = None
        output_text = ProductBrief(**expected).model_dump_json()

    class ResultC:
        final_output = None

        def final_output_as(self, _model):
            return expected

    class ResultD:
        final_output = False

        def final_output_as(self, _model):
            return expected

    assert orch._extract_raw_output(ResultA(), ProductBrief) == expected
    assert orch._extract_raw_output(ResultB(), ProductBrief) is not None
    assert orch._extract_raw_output(ResultC(), ProductBrief) == expected
    assert orch._extract_raw_output(ResultD(), ProductBrief) == expected


def test_run_role_retries_once_on_invalid_output_and_recovers():
    orch = PipelineOrchestrator(build_default_knowledge_provider())
    calls = {"count": 0}

    class FakeResult:
        def __init__(self, final_output=None):
            self.final_output = final_output

    class FakeRunner:
        @staticmethod
        async def run(agent, prompt):
            calls["count"] += 1
            if calls["count"] == 1:
                return FakeResult(final_output=False)  # inválido -> fuerza retry
            return FakeResult(
                final_output={
                    "product_summary": "ok",
                    "business_goals": ["g"],
                    "personas": ["p"],
                    "epics": ["e"],
                    "risks": ["r"],
                }
            )

    previous_runner = orchestrator_module.Runner
    orchestrator_module.Runner = FakeRunner
    try:
        async def _run():
            return await orch._run_role(
                role="product_owner",
                agent_builder=lambda _: object(),
                input_payload={"title": "t", "goal": "g", "constraints": [], "context": {}},
            )

        result = asyncio.run(_run())
        assert result.product_summary == "ok"
        assert calls["count"] == 2
    finally:
        orchestrator_module.Runner = previous_runner


def test_run_role_fails_after_retry_if_output_stays_invalid():
    orch = PipelineOrchestrator(build_default_knowledge_provider())

    class FakeResult:
        def __init__(self, final_output=None):
            self.final_output = final_output

    class FakeRunner:
        @staticmethod
        async def run(agent, prompt):
            return FakeResult(final_output=False)

    previous_runner = orchestrator_module.Runner
    orchestrator_module.Runner = FakeRunner
    try:
        async def _run():
            return await orch._run_role(
                role="product_owner",
                agent_builder=lambda _: object(),
                input_payload={"title": "t", "goal": "g", "constraints": [], "context": {}},
            )

        try:
            asyncio.run(_run())
        except RuntimeError as exc:
            assert "product_owner" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError after retry exhaustion")
    finally:
        orchestrator_module.Runner = previous_runner


def test_optional_clarification_flow_hook_is_available_and_stable():
    orch = PipelineOrchestrator(build_default_knowledge_provider())
    state = ProjectState(
        project_id="p1",
        title="t",
        goal="g" * 10,
        current_phase="functional_analyst",
        constraints=[],
        context={},
        artifacts=ProjectArtifacts(),
        audit_log=[],
    )

    payload = {"k": "v"}
    result = asyncio.run(
        orch._run_optional_clarification_flow(
            role="functional_analyst",
            input_payload=payload,
            state=state,
            max_questions=2,
        )
    )

    assert result == payload
    assert any("Clarification flow evaluado" in line for line in state.audit_log)
