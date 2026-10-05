"""The FastAPI app: routes under ``/api/public/v1``. Everything is a GET but
``POST /predict``, which costs the reader's cases, ``POST /simulate``, which
queues a simulation, ``POST /workloads``, which keeps an uploaded trace, and
``DELETE /simulations/{id}``. The prediction or simulation they leave is read
through the Analyzer's own routes, forwarded read-only under ``/analyzer``.
The three ``POST`` routes are rate limited per client address, counting only
the requests they accept."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from public_api import predict as timing
from public_api import simulate, workloads
from public_api.access_log import AccessLog
from public_api.deployments import BadMember, UnknownDeployment
from public_api.kernels import BadQuery, KernelLibrary, UnknownConfig, UnknownKind
from public_api.limits import RateLimiter

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


class SimulateRequest(BaseModel):
    """One member of a sim preset, named by its axis values
    (``/simulations/presets``), and the workload to run on it."""

    preset: str
    params: dict[str, Any]
    workload: simulate.Workload = Field(default_factory=simulate.Workload)


# How long a prediction stays readable for Read more.
KEEP_PREDICTIONS_S = 6 * 3600
# Requests per client address per minute.
PREDICT_PER_MINUTE = 30
SIMULATE_PER_MINUTE = 6
UPLOAD_PER_MINUTE = 6


def create_app(
    kernels: KernelLibrary,
    runs_dir: Path | None = None,
    analyzer: str | None = None,
    simulations: simulate.SimulationService | None = None,
    *,
    predict_limit: RateLimiter | None = None,
    simulate_limit: RateLimiter | None = None,
    upload_limit: RateLimiter | None = None,
    access_log: AccessLog | None = None,
) -> FastAPI:
    """``runs_dir`` holds the predictions ``POST /predict`` leaves, and
    ``analyzer`` is the base URL of an ``analyze serve`` over it and over the
    simulations' directory; without them the service answers neither.
    ``simulations`` serves the sim presets, runs and workloads; without it the
    ``/simulat*`` and ``/workloads`` routes answer 503. ``access_log``, if
    given, records every request."""
    index = kernels.index
    sources = kernels.sources
    # Where this host keeps predictions and simulations, as the Analyzer may
    # spell them, with the separator: forwarded reports drop it.
    roots = [runs_dir] if runs_dir is not None else []
    if simulations is not None:
        roots.append(simulations.queue.runs_dir)
    hidden_roots = sorted(
        {f"{spelled}/".encode() for root in roots for spelled in (root, root.resolve())},
        key=len,
        reverse=True,
    )
    predict_limit = predict_limit or RateLimiter(PREDICT_PER_MINUTE, 60.0)
    simulate_limit = simulate_limit or RateLimiter(SIMULATE_PER_MINUTE, 60.0)
    upload_limit = upload_limit or RateLimiter(UPLOAD_PER_MINUTE, 60.0)

    @contextmanager
    def limited(limiter: RateLimiter, request: Request) -> Iterator[None]:
        """Count the request against its client's limit, unless the route
        rejects it: a 4xx answer is no work done."""
        client = request.client.host if request.client else "unknown"
        wait = limiter.admit(client)
        if wait is not None:
            raise HTTPException(
                429,
                f"at most {limiter.limit} requests per {limiter.window_s:.0f} s; "
                f"retry in {wait:.0f} s",
                headers={"Retry-After": str(max(1, round(wait)))},
            )
        try:
            yield
        except HTTPException as error:
            if error.status_code < 500:
                limiter.refund(client)
            raise

    app = FastAPI(
        title="ServingStudio public API",
        docs_url=f"{PREFIX}/docs",
        openapi_url=f"{PREFIX}/openapi.json",
    )
    # A kind's rows run to hundreds of kilobytes of JSON.
    app.add_middleware(GZipMiddleware, minimum_size=4096)
    if access_log is not None:
        access_log.install(app)

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
        except (BadQuery, timing.BadCases, workloads.BadWorkload) as error:
            # A request too long for its pool or arch also gives the limit.
            too_long = getattr(error, "too_long", None)
            detail = {"message": str(error), "too_long": too_long} if too_long else str(error)
            raise HTTPException(400, detail) from None
        except (timing.NotPredictable, simulate.NotRunnable) as error:
            raise HTTPException(409, str(error)) from None
        except simulate.UnknownSimulation as error:
            raise HTTPException(404, f"no simulation {error.args[0]!r}") from None
        except simulate.QueueFull as error:
            raise HTTPException(429, str(error), headers={"Retry-After": "60"}) from None

    @app.get(f"{PREFIX}/health")
    async def health() -> dict:
        return {"status": "ok", "sim_commit": kernels.sim_commit}

    @app.get(f"{PREFIX}/models")
    async def models() -> dict:
        """Every checkpoint with its public presets, their axes and members."""
        return await answer(index.catalog)

    @app.get(f"{PREFIX}/models/{{checkpoint}}/{{arch}}/tree")
    async def tree(checkpoint: str, arch: str, request: Request) -> dict:
        """One member's kernels, by section and slot; the query names each axis."""
        params = dict(request.query_params)
        return await answer(index.tree, f"{checkpoint}/{arch}", params)

    @app.post(f"{PREFIX}/predict")
    async def predict(body: PredictRequest, request: Request) -> dict:
        """Each case's time on one member, per section and per tree node."""

        request.state.note = {
            "preset": body.preset,
            "params": body.params,
            "cases": len(body.cases),
            "analyze": body.analyze,
        }
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

        with limited(predict_limit, request):
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

    def need_simulations() -> simulate.SimulationService:
        if simulations is None:
            raise HTTPException(503, "this service runs no simulations")
        return simulations

    @app.get(f"{PREFIX}/simulations/presets")
    async def simulation_presets() -> dict:
        """Every sim preset: its deployment, pools, axes, the captures its
        members replay, and per member what it runs and why it cannot."""
        return await answer(need_simulations().presets)

    @app.post(f"{PREFIX}/simulate", status_code=202)
    async def start_simulation(body: SimulateRequest, request: Request) -> dict:
        """Queue one simulation; read it with ``GET /simulations/{id}``."""
        request.state.note = {
            "preset": body.preset,
            "params": body.params,
            "source": body.workload.source,
        }
        service = need_simulations()
        with limited(simulate_limit, request):
            started = await answer(service.start, body.preset, body.params, body.workload)
        request.state.note["simulation_id"] = started.get("simulation_id")
        return started

    @app.get(f"{PREFIX}/workloads")
    async def workload_sources() -> dict:
        """Where a simulation's requests can come from: a capture, a generator
        (its arguments) or an upload (the formats it may declare)."""
        return await answer(need_simulations().describe_workloads)

    @app.post(f"{PREFIX}/workloads", status_code=201)
    async def upload_workload(
        request: Request,
        input_file_format: str | None = Query(None, alias="format"),
        tags: str | None = None,
    ) -> dict:
        """Keep an uploaded trace (the request body, a CSV) for a day; the
        answer's ``workload_id`` is a simulation's ``workload.upload``. With no
        ``format``, its format and tags are the ones its header fits."""
        service = need_simulations()
        with limited(upload_limit, request):
            body = bytearray()
            async for chunk in request.stream():
                body += chunk
                if len(body) > workloads.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"an upload is at most {workloads.MAX_UPLOAD_BYTES} bytes"
                    )
            request.state.note = {"bytes": len(body), "format": input_file_format}
            names = None if tags is None else [tag for tag in tags.split(",") if tag]
            kept = await answer(service.workloads.upload, bytes(body), input_file_format, names)
        request.state.note["workload_id"] = kept.get("workload_id")
        return kept

    @app.get(f"{PREFIX}/simulations/{{simulation_id}}")
    async def simulation(simulation_id: str) -> dict:
        """Its status; once done, its summary and the Analyzer's run id."""
        return await answer(need_simulations().queue.get, simulation_id)

    @app.delete(f"{PREFIX}/simulations/{{simulation_id}}")
    async def delete_simulation(simulation_id: str) -> dict:
        """Cancel it if it has not finished, and remove it."""
        await answer(need_simulations().queue.delete, simulation_id)
        return {"simulation_id": simulation_id, "status": "deleted"}

    async def forward(kind: str, resource_id: str, subject: str, request: Request) -> Response:
        if analyzer is None:
            raise HTTPException(503, "this service runs no Analyzer")
        if not _SUBJECT.fullmatch(subject) or ".." in subject.split("/"):
            raise HTTPException(404, f"no {kind} route {subject!r}")
        url = f"{analyzer}/api/analyzer/v1/{kind}/{resource_id}/{subject}"
        async with httpx.AsyncClient(timeout=120) as client:
            answer = await client.get(url, params=request.query_params)
        content = answer.content
        # Some reports name the run's directory; keep only its own name.
        for root in hidden_roots:
            content = content.replace(root, b"")
        return Response(
            content,
            status_code=answer.status_code,
            media_type=answer.headers.get("content-type"),
        )

    @app.get(f"{PREFIX}/analyzer/predictions/{{prediction_id}}/{{subject:path}}")
    async def prediction_subject(prediction_id: str, subject: str, request: Request) -> Response:
        """The Analyzer's routes for one prediction (``/api/analyzer/v1/predictions/
        {id}/...``): its descriptor, reports and payloads, read only. No catalog:
        a reader opens the prediction they made."""
        return await forward("predictions", prediction_id, subject, request)

    @app.get(f"{PREFIX}/analyzer/runs/{{run_id}}/{{subject:path}}")
    async def run_subject(run_id: str, subject: str, request: Request) -> Response:
        """The Analyzer's routes for one simulation's run (``/api/analyzer/v1/runs/
        {id}/...``: descriptor, summary, reports, payloads), read only. The run
        id is the ``run_id`` of ``GET /simulations/{id}``. No catalog."""
        return await forward("runs", run_id, subject, request)

    @app.get(f"{PREFIX}/analyzer/kernel-kinds")
    async def kernel_kinds() -> dict:
        """The Analyzer's ``/api/analyzer/v1/kernel-kinds``, read here from the
        same DOCs: each kernel kind's title and category, by which a result
        names and groups its kernels."""
        return await answer(kernels.kinds)

    return app
