from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple, Type

from pydantic import BaseModel

from .agents import (
    ROLE_OUTPUT_MODELS,
    build_backend_agent,
    build_commit_manager_agent,
    build_frontend_agent,
    build_functional_analyst_agent,
    build_product_owner_agent,
    build_qa_agent,
    build_architecture_reviewer_agent,
)
from .knowledge import KnowledgeProvider
from .schemas import ProjectArtifacts, ProjectCreateRequest, ProjectResponse, ProjectState

try:
    from agents import Runner
except Exception:
    Runner = None  # type: ignore

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ==========================================================
# AGENT QUERY GOVERNANCE
# ==========================================================

AGENT_QUERY_GOVERNANCE: Dict[str, Any] = {
    "allowed_sender_roles": {
        "product_owner",
        "functional_analyst",
        "backend_developer",
        "frontend_developer",
        "architecture_reviewer",
        "qa_analyst",
        "commit_manager",
    },
    "allowed_receiver_roles": {
        "product_owner",
        "functional_analyst",
        "backend_developer",
        "frontend_developer",
        "architecture_reviewer",
        "qa_analyst",
        "commit_manager",
    },
    "allowed_categories_by_relation": {
        ("functional_analyst", "product_owner"): {
            "scope_clarification",
            "business_rule_clarification",
        },
        ("backend_developer", "functional_analyst"): {
            "api_contract_clarification",
            "data_rule_clarification",
        },
        ("frontend_developer", "functional_analyst"): {
            "ux_flow_clarification",
            "acceptance_criteria_clarification",
        },
        ("qa_analyst", "functional_analyst"): {
            "testability_clarification",
            "acceptance_criteria_clarification",
        },
        ("qa_analyst", "backend_developer"): {
            "integration_test_clarification",
            "error_handling_clarification",
        },
        ("qa_analyst", "frontend_developer"): {
            "e2e_flow_clarification",
            "state_behavior_clarification",
        },
        ("architecture_reviewer", "backend_developer"): {
            "risk_clarification",
            "architecture_consistency",
        },
        ("architecture_reviewer", "frontend_developer"): {
            "risk_clarification",
            "architecture_consistency",
        },
        ("commit_manager", "qa_analyst"): {
            "release_gate_clarification",
            "residual_risk_clarification",
        },
    },
    # Gobernanza explícita: solo se admite 1 o 2 preguntas por etapa.
    "max_questions_per_stage": 2,
}


class PipelineOrchestrator:
    def __init__(self, knowledge_provider: KnowledgeProvider):
        self.knowledge = knowledge_provider

    async def run_project(self, request: ProjectCreateRequest) -> ProjectResponse:
        project_id = str(uuid.uuid4())

        state = ProjectState(
            project_id=project_id,
            title=request.title,
            goal=request.goal,
            current_phase="product_owner",
            constraints=request.constraints,
            context=request.context,
            artifacts=ProjectArtifacts(),
            audit_log=["Proyecto creado"],
        )

        # ---------------- PRODUCT OWNER ----------------
        product_brief = await self._run_role(
            role="product_owner",
            agent_builder=build_product_owner_agent,
            input_payload={
                "title": request.title,
                "goal": request.goal,
                "constraints": request.constraints,
                "context": request.context,
            },
        )
        state.artifacts.product_brief = product_brief
        state.audit_log.append("Product Owner completado")

        # ---------------- ANALISTA ----------------
        requirements_spec = await self._run_role(
            role="functional_analyst",
            agent_builder=build_functional_analyst_agent,
            input_payload=product_brief.model_dump(),
        )
        state.artifacts.requirements_spec = requirements_spec
        state.audit_log.append("Analista Funcional completado")

        # ---------------- BACKEND ----------------
        backend_spec = await self._run_role(
            role="backend_developer",
            agent_builder=build_backend_agent,
            input_payload={
                "product_brief": product_brief.model_dump(),
                "requirements_spec": requirements_spec.model_dump(),
            },
        )
        state.artifacts.backend_spec = backend_spec
        state.audit_log.append("Backend Developer completado")

        # ---------------- FRONTEND ----------------
        frontend_spec = await self._run_role(
            role="frontend_developer",
            agent_builder=build_frontend_agent,
            input_payload={
                "product_brief": product_brief.model_dump(),
                "requirements_spec": requirements_spec.model_dump(),
                "backend_spec": backend_spec.model_dump(),
            },
        )
        state.artifacts.frontend_spec = frontend_spec
        state.audit_log.append("Frontend Developer completado")

                # ---------------- ARCHITECTURE REVIEW ----------------
        architecture_review = await self._run_role(
            role="architecture_reviewer",
            agent_builder=build_architecture_reviewer_agent,
            input_payload={
                "product_brief": product_brief.model_dump(),
                "requirements_spec": requirements_spec.model_dump(),
                "backend_spec": backend_spec.model_dump(),
                "frontend_spec": frontend_spec.model_dump(),
            },
        )
        state.artifacts.architecture_review = architecture_review
        state.audit_log.append("Architecture Reviewer completado")

        # ---------------- QA ----------------
        test_plan = await self._run_role(
            role="qa_analyst",
            agent_builder=build_qa_agent,
            input_payload={
                "requirements_spec": requirements_spec.model_dump(),
                "backend_spec": backend_spec.model_dump(),
                "frontend_spec": frontend_spec.model_dump(),
                "architecture_review": architecture_review.model_dump(),
            },
        )
        state.artifacts.test_plan = test_plan
        state.audit_log.append("QA Analyst completado")

        # ---------------- RELEASE ----------------
        release_bundle = await self._run_role(
            role="commit_manager",
            agent_builder=build_commit_manager_agent,
            input_payload={
                "requirements_spec": requirements_spec.model_dump(),
                "backend_spec": backend_spec.model_dump(),
                "frontend_spec": frontend_spec.model_dump(),
                "test_plan": test_plan.model_dump(),
                "architecture_review": architecture_review.model_dump(),
            },
        )
        state.artifacts.release_bundle = release_bundle
        state.audit_log.append("Commit Manager completado")

        state.current_phase = "completed"
        state.audit_log.append("Pipeline completado")

        self._persist_state(state)

        return ProjectResponse(
            project_id=project_id,
            status="completed",
            state=state,
        )

    # ==========================================================
    # CORE EXECUTION
    # ==========================================================

    async def _run_role(
        self,
        role: str,
        agent_builder,
        input_payload: Dict[str, Any],
    ) -> BaseModel:

        if Runner is None:
            raise RuntimeError(
                "No se pudo importar `Runner` del Agents SDK. Revisá dependencias."
            )

        print(f"[START] role={role}")

        output_model: Type[BaseModel] = ROLE_OUTPUT_MODELS[role]

        role_context = self.knowledge.get_context(role)
        agent = agent_builder(role_context)

        prompt = self._build_prompt(
            role=role,
            input_payload=input_payload,
            output_model=output_model,
        )

        result = await Runner.run(agent, prompt)

        # Manejo robusto de salida
        raw_output = getattr(result, "final_output", None)

        if raw_output is None:
            raw_output = getattr(result, "final_output_as", None)

        if raw_output is None:
            raise RuntimeError(f"El agente {role} no devolvió output válido")

        parsed = self._coerce_output(output_model, raw_output)

        print(f"[END] role={role}")

        return parsed

    # ==========================================================
    # PROMPT BUILDER
    # ==========================================================

    def _build_prompt(
        self,
        role: str,
        input_payload: Dict[str, Any],
        output_model: Type[BaseModel],
    ) -> str:

        schema = output_model.model_json_schema()

        return (
            f"Rol actual: {role}\n\n"
            f"Input:\n{json.dumps(input_payload, ensure_ascii=False, indent=2)}\n\n"
            f"Respondé SOLO JSON válido compatible con este schema:\n"
            f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
        )

    # ==========================================================
    # OUTPUT PARSER
    # ==========================================================

    def _coerce_output(
        self,
        output_model: Type[BaseModel],
        raw_output: Any,
    ) -> BaseModel:

        if isinstance(raw_output, output_model):
            return raw_output

        if isinstance(raw_output, dict):
            return output_model.model_validate(raw_output)

        if isinstance(raw_output, str):
            return output_model.model_validate_json(raw_output)

        raise TypeError(f"Formato de salida no soportado: {type(raw_output)!r}")

    # ==========================================================
    # AGENT-TO-AGENT QUERY GOVERNANCE
    # ==========================================================

    def _validate_agent_query(
        self,
        from_role: str,
        to_role: str,
        category: str,
        stage_counter: int,
    ) -> Tuple[bool, str]:
        max_questions = AGENT_QUERY_GOVERNANCE["max_questions_per_stage"]
        if max_questions not in (1, 2):
            return False, "Configuración inválida: max_questions_per_stage debe ser 1 o 2"

        if from_role not in AGENT_QUERY_GOVERNANCE["allowed_sender_roles"]:
            return False, f"Rol emisor no permitido: {from_role}"

        if to_role not in AGENT_QUERY_GOVERNANCE["allowed_receiver_roles"]:
            return False, f"Rol receptor no permitido: {to_role}"

        relation = (from_role, to_role)
        allowed_categories = AGENT_QUERY_GOVERNANCE["allowed_categories_by_relation"].get(
            relation
        )
        if not allowed_categories:
            return False, f"Relación no permitida: {from_role} -> {to_role}"

        if category not in allowed_categories:
            return (
                False,
                f"Categoría no permitida para {from_role} -> {to_role}: {category}",
            )

        if stage_counter >= max_questions:
            return (
                False,
                f"Máximo de preguntas por etapa excedido ({max_questions})",
            )

        return True, "ok"

    def _handle_invalid_agent_query(
        self,
        audit_log: list[str],
        stage_artifact: Dict[str, Any],
        from_role: str,
        to_role: str,
        category: str,
        question: str,
        reason: str,
    ) -> None:
        audit_log.append(
            (
                "Consulta bloqueada por gobernanza "
                f"({from_role}->{to_role}, category={category}): {reason}. "
                f"Duda convertida a open_question."
            )
        )
        stage_artifact.setdefault("open_questions", []).append(question)

    def _register_agent_query(
        self,
        audit_log: list[str],
        stage_artifact: Dict[str, Any],
        from_role: str,
        to_role: str,
        category: str,
        question: str,
        stage_counter: int,
    ) -> bool:
        """
        Evalúa la gobernanza de una consulta entre agentes.
        Devuelve True si la consulta está permitida y puede ejecutarse.
        Si no está permitida, registra audit_log y convierte la duda en open_question.
        """
        is_valid, reason = self._validate_agent_query(
            from_role=from_role,
            to_role=to_role,
            category=category,
            stage_counter=stage_counter,
        )
        if not is_valid:
            self._handle_invalid_agent_query(
                audit_log=audit_log,
                stage_artifact=stage_artifact,
                from_role=from_role,
                to_role=to_role,
                category=category,
                question=question,
                reason=reason,
            )
            return False

        return True

    # ==========================================================
    # STORAGE
    # ==========================================================

    def _persist_state(self, state: ProjectState) -> None:
        out_path = DATA_DIR / f"{state.project_id}.json"
        out_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")


# ==============================================================
# LOAD EXISTING PROJECT
# ==============================================================

def load_project_state(project_id: str) -> Tuple[bool, Dict[str, Any]]:
    path = DATA_DIR / f"{project_id}.json"

    if not path.exists():
        return False, {"message": "Proyecto no encontrado"}

    return True, json.loads(path.read_text(encoding="utf-8"))
