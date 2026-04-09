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