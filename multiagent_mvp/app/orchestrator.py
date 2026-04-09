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

# Flujos permitidos para consultas/aclaraciones entre roles.
# Se define explícitamente para evitar NameError si algún flujo de ejecución
# (actual o futuro) lo referencia de forma dinámica.
CLARIFICATION_FLOWS = {
    "functional_analyst": {"product_owner"},
    "backend_developer": {"functional_analyst"},
    "frontend_developer": {"functional_analyst", "backend_developer"},
    "qa_analyst": {"architecture_reviewer"},
    "architecture_reviewer": {
        "product_owner",
        "functional_analyst",
        "backend_developer",
        "frontend_developer",
        "qa_analyst",
    },
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
        if self._should_block_on_architecture_review(architecture_review.approval_status):
            state.current_phase = "blocked_architecture_review"
            state.audit_log.append(
                f"Pipeline bloqueado por Architecture Reviewer: {architecture_review.approval_status}"
            )
            self._persist_state(state)
            return ProjectResponse(
                project_id=project_id,
                status="blocked",
                state=state,
            )

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
        if self._has_blocking_release_gates(test_plan.release_gates):
            state.current_phase = "blocked_qa_gate"
            state.audit_log.append("Pipeline bloqueado por QA release gates bloqueantes")
            self._persist_state(state)
            return ProjectResponse(
                project_id=project_id,
                status="blocked",
                state=state,
            )

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

        scope_allowed, scope_reason = self._enforce_scope_policy(
            role=role,
            input_payload=input_payload,
        )
        if not scope_allowed:
            raise RuntimeError(f"Scope policy violation in {role}: {scope_reason}")

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

    def _extract_agent_dialogue(self, result: Any) -> list[str]:
        """Extrae trazas de diálogo del resultado del SDK sin romper compatibilidad.

        Algunas versiones/flujos del runtime pueden intentar acceder a este helper.
        Si el resultado no expone mensajes estructurados, devolvemos lista vacía.
        """
        if result is None:
            return []

        candidates = (
            getattr(result, "messages", None),
            getattr(result, "new_messages", None),
            getattr(result, "output", None),
        )

        for candidate in candidates:
            if not candidate:
                continue
            if isinstance(candidate, list):
                return [str(item) for item in candidate]
            return [str(candidate)]

        return []

    # ==========================================================
    # STORAGE
    # ==========================================================

    def _persist_state(self, state: ProjectState) -> None:
        out_path = DATA_DIR / f"{state.project_id}.json"
        out_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    def _should_block_on_architecture_review(self, approval_status: str) -> bool:
        normalized = approval_status.strip().lower()
        return normalized in {"rejected", "reject", "blocked", "block"}

    def _has_blocking_release_gates(self, release_gates: list[str]) -> bool:
        """Detecta gates bloqueantes con convención explícita.

        Evitamos heurísticas por keywords sueltas (p.ej. "critical"), porque pueden
        aparecer en gates informativos y bloquear falsos positivos.
        """
        blocking_prefixes = ("BLOCKER:", "FAILED:", "REJECTED:")
        return any(gate.strip().upper().startswith(prefix) for gate in release_gates for prefix in blocking_prefixes)

    def _enforce_scope_policy(
        self,
        role: str,
        input_payload: Dict[str, Any] | None = None,
        *,
        product_owner_approved: bool = False,
    ) -> tuple[bool, str]:
        """Guardrail de alcance con compatibilidad hacia atrás.

        Evita AttributeError en runtimes que esperan este método y permite
        evolucionar una política explícita de scope sin romper ejecuciones actuales.
        """
        payload = input_payload or {}
        scope_change_requested = bool(payload.get("scope_change_requested"))

        if role == "product_owner":
            return True, "Scope policy OK (Product Owner)"

        if scope_change_requested and not product_owner_approved:
            return False, "Scope change rejected: Product Owner approval required"

        return True, "Scope policy OK"


# ==============================================================
# LOAD EXISTING PROJECT
# ==============================================================

def load_project_state(project_id: str) -> Tuple[bool, Dict[str, Any]]:
    path = DATA_DIR / f"{project_id}.json"

    if not path.exists():
        return False, {"message": "Proyecto no encontrado"}

    return True, json.loads(path.read_text(encoding="utf-8"))
