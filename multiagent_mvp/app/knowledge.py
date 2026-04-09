from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class KnowledgeProvider:
    """Proveedor mínimo de conocimiento por agente.

    Reemplazable por vector stores, file search o cualquier capa RAG.
    """

    corpora: Dict[str, List[str]]

    def get_context(self, role: str) -> str:
        docs = self.corpora.get(role, [])
        if not docs:
            return "No hay contexto adicional específico para este rol."
        return "\n".join(f"- {item}" for item in docs)


def build_default_knowledge_provider() -> KnowledgeProvider:
    return KnowledgeProvider(
        corpora={
            "product_owner": [
                "Priorizar outcomes de negocio sobre features.",
                "Definir épicas y objetivos medibles.",
                "Mantener trazabilidad entre objetivo, KPI y backlog priorizado.",
            ],
            "functional_analyst": [
                "Cada historia debe incluir criterios de aceptación verificables.",
                "Registrar preguntas abiertas y reglas de negocio.",
                "Eliminar ambigüedad usando lenguaje observable y medible.",
            ],
            "backend_developer": [
                "Priorizar diseño REST claro y separación por capas.",
                "Explicitar entidades, endpoints y riesgos técnicos.",
                "Diseñar seguridad por defecto y estrategia de observabilidad.",
            ],
            "frontend_developer": [
                "Definir pantallas, componentes y estados principales.",
                "Considerar accesibilidad y feedback de usuario.",
                "Modelar estados de error/loading/empty/success en cada flujo crítico.",
            ],
            "architecture_reviewer": [
                "Evaluar consistencia entre requerimientos, backend y frontend.",
                "Priorizar simplificaciones de arquitectura para un MVP.",
                "Balancear atributos de calidad: seguridad, mantenibilidad y rendimiento.",
            ],
            "qa_analyst": [
                "Cubrir happy path, errores y regresión mínima.",
                "Definir gates de release claros.",
                "Priorizar cobertura según criticidad y probabilidad de fallo.",
            ],
            "commit_manager": [
                "Usar Conventional Commits.",
                "Preparar checklist de merge y notas de release.",
                "Asegurar estrategia de rollback y comunicación de riesgos.",
            ],
        }
    )
