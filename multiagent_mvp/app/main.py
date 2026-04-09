from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .knowledge import build_default_knowledge_provider
from .orchestrator import PipelineOrchestrator, load_project_state
from .schemas import ProjectCreateRequest, ProjectResponse

import os


app = FastAPI(
    title="Software Factory Multi-Agent MVP1",
    version="0.1.0",
    description="MVP para orquestar agentes especializados de desarrollo de software con OpenAI.",
)

orchestrator = PipelineOrchestrator(build_default_knowledge_provider())


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return """
    <html>
      <head><title>Multi-Agent MVP</title></head>
      <body style="font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto;">
        <h1>Software Factory Multi-Agent MVP1</h1>
        <p>API disponible en <a href="/docs">/docs</a>.</p>
        <p>Endpoint principal: <code>POST /projects</code></p>
      </body>
    </html>
    """


@app.post("/projects", response_model=ProjectResponse)
async def create_project(payload: ProjectCreateRequest) -> ProjectResponse:
    try:
        return await orchestrator.run_project(payload)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/projects/{project_id}")
async def get_project(project_id: str):
    found, result = load_project_state(project_id)
    if not found:
        raise HTTPException(status_code=404, detail=result["message"])
    return result
