from __future__ import annotations

import os
from textwrap import dedent
from typing import Type

from pydantic import BaseModel

from .schemas import (
    ProductBrief,
    RequirementsSpec,
    BackendSpec,
    FrontendSpec,
    TestPlan,
    ReleaseBundle,
    ArchitectureReviewerReport
)

# ==========================================================
# MODEL CONFIG
# ==========================================================

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

try:
    from agents import Agent
except Exception:
    Agent = None  # type: ignore


# ==========================================================
# AGENT FACTORY
# ==========================================================

ROLE_SKILLS = {
    "product_owner": [
        "Product discovery y definición de alcance MVP",
        "Priorización por valor de negocio y riesgo",
        "Definición de KPIs y outcomes",
        "Gestión de stakeholders y trade-offs",
    ],
    "functional_analyst": [
        "Elicitación y especificación de requerimientos",
        "Historias de usuario y criterios de aceptación",
        "Modelado de reglas de negocio",
        "Detección de ambigüedades y dependencias",
    ],
    "backend_developer": [
        "Diseño de arquitectura backend por capas",
        "Diseño de APIs REST y contratos",
        "Modelado de entidades y persistencia",
        "Seguridad, escalabilidad y observabilidad",
    ],
    "frontend_developer": [
        "Diseño de UX/UI orientado a tareas",
        "Arquitectura de componentes reutilizables",
        "Manejo de estado y flujos de interacción",
        "Accesibilidad y consistencia visual",
    ],
    "architecture_reviewer": [
        "Revisión de consistencia end-to-end",
        "Identificación de sobreingeniería",
        "Evaluación de riesgos técnicos y de seguridad",
        "Recomendaciones de simplificación para MVP",
    ],
    "qa_analyst": [
        "Estrategia de testing funcional y no funcional",
        "Diseño de casos de prueba y cobertura",
        "Definición de quality gates de release",
        "Análisis de regresión y riesgos de calidad",
    ],
    "commit_manager": [
        "Estrategia de branching y versionado",
        "Conventional Commits y atomicidad de cambios",
        "Preparación de release notes y checklist",
        "Coordinación de entrega y handoff a operación",
    ],
}


def _format_skills(role_key: str) -> str:
    skills = ROLE_SKILLS.get(role_key, [])
    if not skills:
        return "- Sin skills definidas."
    return "\n".join(f"- {skill}" for skill in skills)

def _make_agent(name: str, instructions: str, output_model: Type[BaseModel]):
    if Agent is None:
        raise RuntimeError(
            "No se pudo importar `agents`. Ejecutá: pip install -r requirements.txt"
        )

    return Agent(
        name=name,
        instructions=instructions,
        model=MODEL_NAME,
        output_type=output_model,
    )


# ==========================================================
# AGENTES POR ROL
# ==========================================================

def build_product_owner_agent(role_context: str):
    instructions = dedent(
        f"""
        Sos un Product Owner senior.

        Tu responsabilidad es transformar una iniciativa en un product brief claro, estructurado y accionable.

        Reglas:
        - Pensá en términos de negocio
        - Definí objetivos claros
        - Identificá stakeholders y usuarios
        - No agregues texto fuera del JSON

        Skills del rol:
        {_format_skills("product_owner")}

        Contexto:
        {role_context}
        """
    ).strip()

    return _make_agent("Product Owner", instructions, ProductBrief)


def build_functional_analyst_agent(role_context: str):
    instructions = dedent(
        f"""
        Sos un Analista Funcional senior.

        Convertí el product brief en requerimientos verificables.

        Reglas:
        - Generá historias de usuario claras
        - Definí criterios de aceptación
        - Incluí reglas de negocio
        - Evitá ambigüedad

        Skills del rol:
        {_format_skills("functional_analyst")}

        Contexto:
        {role_context}
        """
    ).strip()

    return _make_agent("Functional Analyst", instructions, RequirementsSpec)


def build_backend_agent(role_context: str):
    instructions = dedent(
        f"""
        Sos un Backend Developer senior.

        Diseñá la arquitectura backend.

        Reglas:
        - Definí entidades y relaciones
        - Diseñá endpoints REST claros
        - Considerá seguridad y escalabilidad
        - Identificá riesgos técnicos

        Skills del rol:
        {_format_skills("backend_developer")}

        Contexto:
        {role_context}
        """
    ).strip()

    return _make_agent("Backend Developer", instructions, BackendSpec)


def build_frontend_agent(role_context: str):
    instructions = dedent(
        f"""
        Sos un Frontend Developer senior.

        Diseñá la experiencia de usuario.

        Reglas:
        - Definí pantallas claras
        - Proponé componentes reutilizables
        - Pensá en estados y flujos
        - Considerá accesibilidad

        Skills del rol:
        {_format_skills("frontend_developer")}

        Contexto:
        {role_context}
        """
    ).strip()

    return _make_agent("Frontend Developer", instructions, FrontendSpec)


def build_qa_agent(role_context: str):
    instructions = dedent(
        f"""
        Sos un QA Analyst senior.

        Diseñá un plan de pruebas completo.

        Reglas:
        - Cubrí casos felices y edge cases
        - Definí criterios de aceptación
        - Incluí pruebas funcionales y no funcionales
        - Pensá en regresión

        Skills del rol:
        {_format_skills("qa_analyst")}

        Contexto:
        {role_context}
        """
    ).strip()

    return _make_agent("QA Analyst", instructions, TestPlan)


def build_commit_manager_agent(role_context: str):
    instructions = dedent(
        f"""
        Sos un Commit Manager / Release Manager.

        Prepará la estrategia de release.

        Reglas:
        - Usá Conventional Commits
        - Generá mensajes claros
        - Incluí checklist de release
        - Definí estrategia de branching

        Skills del rol:
        {_format_skills("commit_manager")}

        Contexto:
        {role_context}
        """
    ).strip()

    return _make_agent("Commit Manager", instructions, ReleaseBundle)

def build_architecture_reviewer_agent(role_context: str):
    instructions = dedent(
        f"""
        Sos un Architecture Reviewer senior.

        Tu responsabilidad es revisar la consistencia global entre requerimientos,
        backend y frontend, con foco en MVP, seguridad, mantenibilidad y riesgo técnico.

        Reglas:
        - Detectá sobreingeniería
        - Marcá inconsistencias entre capas
        - Priorizá simplificaciones razonables para un MVP
        - Señalá riesgos de seguridad y compliance
        - No inventes features fuera del alcance
        - Devolvé solo JSON válido

        Skills del rol:
        {_format_skills("architecture_reviewer")}

        Contexto:
        {role_context}
        """
    ).strip()

    return _make_agent("Architecture Reviewer", instructions, ArchitectureReviewerReport)
# ==========================================================
# OUTPUT MODELS MAPPING (CRÍTICO)
# ==========================================================

ROLE_OUTPUT_MODELS = {
    "product_owner": ProductBrief,
    "functional_analyst": RequirementsSpec,
    "backend_developer": BackendSpec,
    "frontend_developer": FrontendSpec,
    "architecture_reviewer": ArchitectureReviewerReport,
    "qa_analyst": TestPlan,
    "commit_manager": ReleaseBundle,
}
