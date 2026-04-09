from __future__ import annotations

import asyncio
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


class ClarificationAnswer(BaseModel):
    answer: str


CLARIFICATION_FLOWS: Dict[str, Dict[str, Any]] = {
    "functional_analyst": {
        "ask_role": "product_owner",
        "topic": "scope/priority",
        "max_queries_per_stage": 1,
        "timeout_seconds": 25,
        "max_retries": 1,
    },
    "backend_developer": {
        "ask_role": "functional_analyst",
        "topic": "business rules",
        "max_queries_per_stage": 1,
        "timeout_seconds": 25,
        "max_retries": 1,
    },
    "frontend_developer": {
        "ask_role": "backend_developer",
        "topic": "API contract",
        "max_queries_per_stage": 1,
        "timeout_seconds": 25,
        "max_retries": 1,
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
            state=state,
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
            state=state,
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
            state=state,
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
        state: ProjectState | None = None,
    ) -> BaseModel:

        if Runner is None:
            raise RuntimeError(
                "No se pudo importar `Runner` del Agents SDK. Revisá dependencias."
            )

        print(f"[START] role={role}")

        output_model: Type[BaseModel] = ROLE_OUTPUT_MODELS[role]

        role_context = self.knowledge.get_context(role)
        agent = agent_builder(role_context)

        consultation_payload = await self._run_optional_clarification_flow(
            role=role,
            input_payload=input_payload,
            state=state,
        )

        prompt = self._build_prompt(
            role=role,
            input_payload=consultation_payload,
            output_model=output_model,
        )

        result = await self._run_with_retry(
            agent=agent,
            prompt=prompt,
            timeout_seconds=45,
            max_retries=1,
            role=role,
        )

        # Manejo robusto de salida
        raw_output = getattr(result, "final_output", None)

        if raw_output is None:
            raw_output = getattr(result, "final_output_as", None)

        if raw_output is None:
            raise RuntimeError(f"El agente {role} no devolvió output válido")

        parsed = self._coerce_output(output_model, raw_output)

        print(f"[END] role={role}")

        return parsed

    async def _run_optional_clarification_flow(
        self,
        role: str,
        input_payload: Dict[str, Any],
        state: ProjectState | None = None,
    ) -> Dict[str, Any]:
        flow = CLARIFICATION_FLOWS.get(role)
        if flow is None:
            return input_payload

        if not self._needs_clarification(input_payload):
            return input_payload

        max_queries = int(flow.get("max_queries_per_stage", 1))
        if max_queries <= 0:
            return input_payload

        ask_role = str(flow["ask_role"])
        topic = str(flow["topic"])
        timeout_seconds = int(flow.get("timeout_seconds", 25))
        max_retries = int(flow.get("max_retries", 1))

        queries_done = 0
        enriched_payload = dict(input_payload)
        clarification_entries: list[dict[str, Any]] = []

        while queries_done < max_queries and self._needs_clarification(enriched_payload):
            query = self._build_clarification_question(
                role=role,
                ask_role=ask_role,
                topic=topic,
                input_payload=enriched_payload,
            )

            answer = await self._request_clarification(
                ask_role=ask_role,
                query=query,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )

            clarification_entries.append(
                {
                    "stage": role,
                    "from_role": role,
                    "to_role": ask_role,
                    "topic": topic,
                    "query": query,
                    "answer": answer,
                }
            )

            queries_done += 1
            enriched_payload["clarifications"] = clarification_entries
            enriched_payload["clarification_context"] = (
                "Usá estas respuestas para resolver ambigüedades antes de generar el artefacto."
            )

        if state is not None and clarification_entries:
            state.agent_dialogue.extend(clarification_entries)
            state.audit_log.append(
                f"{role}: se realizaron {len(clarification_entries)} consulta(s) a {ask_role}"
            )

        return enriched_payload

    async def _request_clarification(
        self,
        ask_role: str,
        query: str,
        timeout_seconds: int,
        max_retries: int,
    ) -> str:
        if ask_role == "product_owner":
            builder = build_product_owner_agent
        elif ask_role == "functional_analyst":
            builder = build_functional_analyst_agent
        elif ask_role == "backend_developer":
            builder = build_backend_agent
        else:
            raise ValueError(f"Rol de consulta no soportado: {ask_role}")

        role_context = self.knowledge.get_context(ask_role)
        agent = builder(role_context)

        consultation_prompt = (
            f"Actuá como {ask_role} y respondé SOLO JSON válido.\n"
            f"Consulta: {query}\n\n"
            f"Schema:\n{json.dumps(ClarificationAnswer.model_json_schema(), ensure_ascii=False, indent=2)}"
        )

        result = await self._run_with_retry(
            agent=agent,
            prompt=consultation_prompt,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            role=f"{ask_role}(consultation)",
        )

        raw_output = getattr(result, "final_output", None)
        if raw_output is None:
            raw_output = getattr(result, "final_output_as", None)
        if raw_output is None:
            raise RuntimeError(f"{ask_role} no devolvió respuesta de consulta")

        parsed = self._coerce_output(ClarificationAnswer, raw_output)
        return parsed.answer

    async def _run_with_retry(
        self,
        agent: Any,
        prompt: str,
        timeout_seconds: int,
        max_retries: int,
        role: str,
    ) -> Any:
        attempt = 0
        while True:
            try:
                return await asyncio.wait_for(
                    Runner.run(agent, prompt),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"Timeout ejecutando {role} tras {attempt} intento(s)"
                    ) from None
            except Exception:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"Error ejecutando {role} tras {attempt} intento(s)"
                    ) from None

    def _needs_clarification(self, input_payload: Dict[str, Any]) -> bool:
        explicit_signal = bool(
            input_payload.get("clarification_needed")
            or input_payload.get("_clarification_needed")
            or input_payload.get("clarification_request")
        )
        if explicit_signal:
            return True

        markers = {"tbd", "todo", "unknown", "pendiente", "por definir", "?"}
        serialized = json.dumps(input_payload, ensure_ascii=False).lower()
        if any(marker in serialized for marker in markers):
            return True

        open_questions = input_payload.get("open_questions")
        if isinstance(open_questions, list) and len(open_questions) > 0:
            return True

        return False

    def _build_clarification_question(
        self,
        role: str,
        ask_role: str,
        topic: str,
        input_payload: Dict[str, Any],
    ) -> str:
        return (
            f"El rol {role} necesita aclaración sobre {topic}. "
            f"Respondé con precisión y foco MVP usando este contexto: "
            f"{json.dumps(input_payload, ensure_ascii=False)}. "
            f"Limitá la respuesta a decisiones accionables para {role}."
        )

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
