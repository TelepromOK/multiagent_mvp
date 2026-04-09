# MVP multiagente para fábrica de software con OpenAI

Este proyecto es un **MVP funcional** para orquestar un equipo de agentes especializados:

- Product Owner
- Analista Funcional
- Backend Developer
- Frontend Developer
- QA Analyst
- Commit Manager

La idea es que cada agente:

1. reciba un input estructurado,
2. consulte una base de conocimiento propia,
3. produzca un output estructurado,
4. y ese output pase al siguiente agente.

## Stack

- **FastAPI** para exponer la API del MVP
- **OpenAI Agents SDK** para definir agentes y handoffs
- **Pydantic** para esquemas estrictos de entrada/salida
- **JSON local** como persistencia mínima de `project_state`

> Este MVP prioriza arquitectura, contratos y trazabilidad. No intenta ser un producto final.

## Arquitectura

```text
Cliente/API
   |
   v
Orchestrator
   |
   +--> Product Owner Agent
   |        |
   |        v
   |   product_brief
   |
   +--> Functional Analyst Agent
   |        |
   |        v
   |   requirements_spec
   |
   +--> Backend Agent
   |        |
   |        v
   |   backend_spec
   |
   +--> Frontend Agent
   |        |
   |        v
   |   frontend_spec
   |
   +--> QA Agent
   |        |
   |        v
   |   test_plan
   |
   +--> Commit Manager Agent
            |
            v
       release_bundle
```

## Requisitos

- Python 3.11+
- Variable de entorno `OPENAI_API_KEY`

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Ejecutar

```bash
uvicorn app.main:app --reload
```

Abrí luego:

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

## Endpoint principal

### `POST /projects`
Crea una iniciativa y ejecuta el pipeline completo.

Ejemplo de payload:

```json
{
  "title": "Portal de onboarding de clientes",
  "goal": "Construir un portal web para alta, seguimiento documental y aprobación de nuevos clientes.",
  "constraints": [
    "Debe exponer API REST",
    "Debe tener autenticación",
    "Debe permitir seguimiento del estado"
  ],
  "context": {
    "industry": "fintech",
    "priority": "alta",
    "deadline": "2026-06-30"
  }
}
```

## Cómo conectar bases de conocimiento por agente

En este MVP hay una capa `knowledge.py` con un `KnowledgeProvider` simple. Ahí podés reemplazar la implementación por:

- OpenAI File Search / vector stores
- un RAG propio
- Postgres + embeddings
- documentos por rol
- repositorios internos por dominio

La interfaz ya está separada para que cada agente consulte una colección distinta.

## Próximos pasos recomendados

1. Persistir artefactos en Postgres.
2. Agregar revisión cruzada entre agentes.
3. Enviar a GitHub/GitLab como PR draft.
4. Conectar file search / vector stores reales.
5. Separar pipeline síncrono de ejecución asíncrona.
6. Agregar frontend de seguimiento.

## Nota sobre OpenAI

Este MVP usa el enfoque que OpenAI recomienda hoy para apps agentic nuevas: **Responses API** como base y **Agents SDK** para agentes, tools y orchestration. La documentación oficial del SDK también describe agentes con instrucciones, tools, guardrails, handoffs y structured outputs. citeturn211769search0turn211769search1turn211769search7turn211769search25
