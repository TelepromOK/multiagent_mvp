from __future__ import annotations

import json
import re
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
        state.current_phase = "architecture_review_gate"

        is_architecture_approved, architecture_block_reasons = self._evaluate_architecture_gate(
            architecture_review
        )
        if not is_architecture_approved:
            state.current_phase = "blocked_architecture_review"
            state.audit_log.append(
                self._format_blocking_audit_entry(
                    gate_name="architecture_review",
                    block_reasons=architecture_block_reasons,
                )
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
        state.current_phase = "qa_gate"

        is_qa_approved, qa_block_reasons = self._evaluate_qa_gate(test_plan)
        if not is_qa_approved:
            state.current_phase = "blocked_qa_gate"
            state.audit_log.append(
                self._format_blocking_audit_entry(
                    gate_name="qa_gate",
                    block_reasons=qa_block_reasons,
                )
            )
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
    # STORAGE
    # ==========================================================

    def _persist_state(self, state: ProjectState) -> None:
        out_path = DATA_DIR / f"{state.project_id}.json"
        out_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    # ==========================================================
    # GATES
    # ==========================================================

    def _evaluate_architecture_gate(
        self, architecture_review: BaseModel
    ) -> Tuple[bool, list[str]]:
        approval_status = (
            getattr(architecture_review, "approval_status", "") or ""
        ).strip().lower()
        issues = getattr(architecture_review, "issues", []) or []
        security_observations = (
            getattr(architecture_review, "security_observations", []) or []
        )

        block_reasons: list[str] = []
        allowed_statuses = {"approved", "approved_with_changes"}
        if approval_status not in allowed_statuses:
            block_reasons.append(
                "approval_status fuera de criterio de continuidad "
                f"(valor={approval_status or 'vacío'}, esperado in {sorted(allowed_statuses)})"
            )

        critical_risks = self._extract_critical_items(issues + security_observations)
        if critical_risks:
            block_reasons.append(
                "riesgos críticos detectados: " + " | ".join(critical_risks)
            )

        return len(block_reasons) == 0, block_reasons

    def _evaluate_qa_gate(self, test_plan: BaseModel) -> Tuple[bool, list[str]]:
        release_gates = getattr(test_plan, "release_gates", []) or []
        coverage = getattr(test_plan, "coverage", []) or []

        block_reasons: list[str] = []
        failed_gates = self._extract_failed_release_gates(release_gates)
        if failed_gates:
            block_reasons.append("release_gates no cumplidos: " + " | ".join(failed_gates))

        coverage_gaps = self._extract_gap_items(coverage + release_gates)
        if coverage_gaps:
            block_reasons.append("gaps de QA/coverage: " + " | ".join(coverage_gaps))

        return len(block_reasons) == 0, block_reasons

    def _extract_critical_items(self, items: list[str]) -> list[str]:
        critical_pattern = re.compile(
            r"\b(critical|crítico|critico|high risk|severo|bloqueante|blocker)\b",
            re.IGNORECASE,
        )
        return [item for item in items if critical_pattern.search(item or "")]

    def _extract_gap_items(self, items: list[str]) -> list[str]:
        gap_pattern = re.compile(
            r"\b(gap|faltante|missing|pendiente|incompleto|insuficiente)\b",
            re.IGNORECASE,
        )
        return [item for item in items if gap_pattern.search(item or "")]

    def _extract_failed_release_gates(self, release_gates: list[str]) -> list[str]:
        failed_pattern = re.compile(
            r"\b(fail|failed|no cumple|blocked|bloqueado|rechazado|pendiente)\b",
            re.IGNORECASE,
        )
        return [gate for gate in release_gates if failed_pattern.search(gate or "")]

    def _format_blocking_audit_entry(
        self, gate_name: str, block_reasons: list[str]
    ) -> str:
        reasons = " || ".join(block_reasons) if block_reasons else "sin detalle"
        return f"Pipeline bloqueado en {gate_name}. Criterios incumplidos: {reasons}"


# ==============================================================
# LOAD EXISTING PROJECT
# ==============================================================

def load_project_state(project_id: str) -> Tuple[bool, Dict[str, Any]]:
    path = DATA_DIR / f"{project_id}.json"

    if not path.exists():
        return False, {"message": "Proyecto no encontrado"}

    return True, json.loads(path.read_text(encoding="utf-8"))
