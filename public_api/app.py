"""The FastAPI app: routes under ``/api/public/v1``, all read-only. Everything is
a GET but ``POST /predict``, which costs the reader's cases; the prediction it
leaves is read through the Analyzer's prediction routes, forwarded under
``/analyzer``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from public_api import predict as timing
from public_api.deployments import BadMember, UnknownDeployment
from public_api.kernels import BadQuery, KernelLibrary, UnknownConfig, UnknownKind

PREFIX = "/api/public/v1"
# A forwarded prediction route's tail: `descriptor`, `subjects/...`, `cases/...`.
_SUBJECT = re.compile(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*")


class PredictRequest(BaseModel):
    """One member of a preset, named by its axis values, and the cases to cost
    on it (``/models``' ``case_fields`` give their shape)."""

    preset: str
    params: dict[str, Any]
    cases: list[Any]
    # Also analyze it and keep it for Read more: the answer names it by
    # `prediction_id` under `/analyzer/predictions/`.
    analyze: bool = False


# How long a prediction stays readable for Read more.
KEEP_PREDICTIONS_S = 6 * 3600


def create_app(
    kernels: KernelLibrary, runs_dir: Path | None = None, analyzer: str | None = None
) -> FastAPI:
    """``runs_dir`` holds the predictions ``POST /predict`` leaves, and
    ``analyzer`` is the base URL of an ``analyze serve`` over it; without them
    the service answers neither."""
    index = kernels.index
    sources = kernels.sources
    app = FastAPI(
        title="ServingStudio public API",
        docs_url=f"{PREFIX}/docs",
        openapi_url=f"{PREFIX}/openapi.json",
    )
    # A kind's rows run to hundreds of kilobytes of JSON.
    app.add_middleware(GZipMiddleware, minimum_size=4096)

    async def answer(build, *args):
        # Documents come from sqlite and simulator subprocesses: keep them off the loop.
        try:
            return await run_in_threadpool(build, *args)
        except UnknownKind as error:
            raise HTTPException(404, f"unknown kernel kind {error.args[0]!r}") from None
        except UnknownConfig as error:
            raise HTTPException(
                404, f"no public deployment asks for config {error.args[0]!r}"
            ) from None
        except UnknownDeployment as error:
            raise HTTPException(404, f"no public preset {error.args[0]!r}") from None
        except BadMember as error:
            raise HTTPException(400, {"message": str(error), "choices": error.choices}) from None
        except (BadQuery, timing.BadCases) as error:
            raise HTTPException(400, str(error)) from None
        except timing.NotPredictable as error:
            raise HTTPException(409, str(error)) from None

    @app.get(f"{PREFIX}/health")
    async def health() -> dict:
        return {"status": "ok", "sim_commit": kernels.sim_commit}

    @app.get(f"{PREFIX}/models")
    async def models() -> dict:
        """Every checkpoint with its public presets, their axes and members."""
        return await answer(index.catalog)

    @app.get(f"{PREFIX}/models/{{checkpoint}}/{{arch}}/tree")
    async def tree(checkpoint: str, arch: str, request: Request) -> dict:
        """One member's cost tree, structure only; the query names each axis."""
        params = dict(request.query_params)
        return await answer(index.tree, f"{checkpoint}/{arch}", params)

    @app.post(f"{PREFIX}/predict")
    async def predict(body: PredictRequest) -> dict:
        """Each case's time on one member, per section and per tree node."""

        if runs_dir is None:
            raise HTTPException(503, "this service keeps no runs directory")

        def run() -> dict:
            _, member = index.member(body.preset, body.params)
            if body.analyze:
                timing.prune(runs_dir, KEEP_PREDICTIONS_S)
            result = timing.predict(
                runs_dir, member, body.cases, sources.build_type, analyze=body.analyze
            )
            return {"sim_commit": index.sim_commit, **result}

        return await answer(run)

    @app.get(f"{PREFIX}/kernels")
    async def catalog() -> dict:
        return await answer(kernels.catalog)

    @app.get(f"{PREFIX}/kernels/{{kind}}")
    async def kernel(kind: str) -> dict:
        return await answer(kernels.kernel, kind)

    @app.get(f"{PREFIX}/kernels/{{kind}}/rows")
    async def rows(kind: str, request: Request):
        """Measured rows. Every query parameter except ``format`` filters a column
        by equality: ``gpu``, ``backend`` or any argument of the kind."""

        filters = dict(request.query_params)
        fmt = filters.pop("format", "json")
        if fmt not in ("json", "csv"):
            raise HTTPException(400, "format must be json or csv")
        document = await answer(kernels.rows, kind, filters)
        if fmt == "csv":
            return PlainTextResponse(
                KernelLibrary.rows_csv(document),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{kind}.csv"'},
            )
        return document

    @app.get(f"{PREFIX}/kernels/{{kind}}/configs")
    async def configs(kind: str) -> dict:
        return await answer(kernels.configs, kind)

    @app.get(f"{PREFIX}/kernels/{{kind}}/configs/{{config_id}}")
    async def config(kind: str, config_id: str) -> dict:
        return await answer(kernels.config, kind, config_id)

    @app.get(f"{PREFIX}/analyzer/predictions/{{prediction_id}}/{{subject:path}}")
    async def prediction_subject(prediction_id: str, subject: str, request: Request) -> Response:
        """The Analyzer's routes for one prediction (``/api/analyzer/v1/predictions/
        {id}/...``): its descriptor, reports and payloads, read only. No catalog:
        a reader opens the prediction they made."""
        if analyzer is None:
            raise HTTPException(503, "this service runs no Analyzer")
        if not _SUBJECT.fullmatch(subject) or ".." in subject.split("/"):
            raise HTTPException(404, f"no prediction route {subject!r}")
        url = f"{analyzer}/api/analyzer/v1/predictions/{prediction_id}/{subject}"
        async with httpx.AsyncClient(timeout=120) as client:
            answer = await client.get(url, params=request.query_params)
        return Response(
            answer.content,
            status_code=answer.status_code,
            media_type=answer.headers.get("content-type"),
        )

    return app
