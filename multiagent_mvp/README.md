# MVP multiagente para fábrica de software con OpenAI

Este proyecto es un **MVP funcional** para orquestar un equipo de agentes especializados:

- Product Owner
- Analista Funcional
- Backend Developer
- Frontend Developer
- Architecture Reviewer
- QA Analyst
- Commit Manager

La idea es que cada agente:

1. reciba un input estructurado,
2. consulte una base de conocimiento propia,
3. aplique skills específicos según su rol,
4. produzca un output estructurado,
5. y ese output pase al siguiente agente.

## Stack

- **FastAPI** para exponer la API del MVP
- **OpenAI Agents SDK** para definir agentes y handoffs
- **Pydantic** para esquemas estrictos de entrada/salida
- **JSON local** como persistencia mínima de `project_state`

> Este MVP prioriza arquitectura, contratos y trazabilidad. No intenta ser un producto final.

## Qué es este sistema (en esencia)

Este repo implementa una **Software Factory autónoma basada en agentes de IA**.  
No está pensado como un chatbot: está pensado como un pipeline que transforma una necesidad en artefactos de ingeniería listos para construir software.

Flujo conceptual:

1. interpreta una necesidad,
2. la transforma en especificaciones,
3. diseña una solución completa,
4. la valida con una revisión independiente,
5. y prepara su entrega.

En una línea: este MVP ya es un **AI Software Design Pipeline**.

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
   +--> Architecture Reviewer Agent
   |        |
   |        v
   |   architecture_review
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

## Patrón arquitectónico implementado

El núcleo del sistema es un **pipeline multiagente secuencial con contratos tipados**:

- cada etapa está desacoplada de la siguiente,
- cada agente tiene responsabilidad explícita por rol,
- el output de cada etapa está validado por esquemas (JSON tipado),
- el estado global se persiste en `project_state`,
- y se mantiene trazabilidad completa en `audit_log`.

Esto habilita re-ejecución controlada, auditoría y evolución incremental.

## Qué produce (y qué no)

### Produce artefactos de ingeniería

- Product Brief
- Requirements Spec
- Backend Spec
- Frontend Spec
- Architecture Review
- Test Plan
- Release Bundle

### Todavía no produce

- código de aplicación listo para producción de forma autónoma,
- CI/CD completamente autónomo,
- loops de autocorrección end-to-end.

## Etapa actual y evolución

Este MVP está en **Fase 1: Software Design Automation**.

Próxima frontera: **Fase 2: Software Delivery Automation**, donde la evolución natural es:

1. generación de código backend/frontend,
2. ejecución automática de tests,
3. loop QA → Dev con realimentación,
4. integración total con Git (commits/PRs automáticos),
5. operación como equipo de desarrollo autónomo.

## Diferenciales clave del enfoque

1. **Separación por roles**: simula una organización real de software.
2. **Outputs estructurados**: usa contratos JSON en lugar de texto libre.
3. **Reviewer independiente**: incluye una etapa de cuestionamiento arquitectónico antes de QA/release.

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
