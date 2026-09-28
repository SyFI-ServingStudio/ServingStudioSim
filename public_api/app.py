"""The FastAPI app: routes under ``/api/public/v1``, all GET, all read-only."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse

from public_api.arch.library import ArchLibrary, BadParams, UnknownArch, UnsupportedParams
from public_api.kernel.library import BadQuery, KernelLibrary, UnknownConfig, UnknownKind

PREFIX = "/api/public/v1"


def create_app(kernels: KernelLibrary, archs: ArchLibrary | None = None) -> FastAPI:
    archs = archs or ArchLibrary(kernels)
    app = FastAPI(
        title="ServingStudio public API",
        docs_url=f"{PREFIX}/docs",
        openapi_url=f"{PREFIX}/openapi.json",
    )
    # A cost tree and a config grid run to hundreds of kilobytes of JSON.
    app.add_middleware(GZipMiddleware, minimum_size=4096)

    async def answer(build, *args):
        # Documents come from sqlite and simulator subprocesses: keep them off the loop.
        try:
            return await run_in_threadpool(build, *args)
        except UnknownKind as error:
            raise HTTPException(404, f"unknown kernel kind {error.args[0]!r}") from None
        except UnknownConfig as error:
            raise HTTPException(404, f"unknown kernel config {error.args[0]!r}") from None
        except BadQuery as error:
            raise HTTPException(400, str(error)) from None
        except UnknownArch as error:
            raise HTTPException(404, f"unknown arch {error.args[0]!r}") from None
        except BadParams as error:
            raise HTTPException(400, {"message": str(error), "choices": error.choices}) from None
        except UnsupportedParams as error:
            raise HTTPException(404, {"message": str(error), "choices": error.choices}) from None

    @app.get(f"{PREFIX}/health")
    async def health() -> dict:
        return {"status": "ok", "sim_commit": kernels.sim_commit}

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

    @app.get(f"{PREFIX}/kernels/{{kind}}/configs/{{config_hash}}")
    async def config(kind: str, config_hash: str, gpu: str | None = None) -> dict:
        """One config's grid on the simulator's cache axes, joined to the rows.
        ``gpu`` is needed only when the config is registered on more than one."""

        return await answer(kernels.config, kind, config_hash, gpu)

    @app.get(f"{PREFIX}/archs")
    async def arch_catalog() -> dict:
        return await answer(archs.catalog)

    @app.get(f"{PREFIX}/archs/{{arch}}")
    async def arch(arch: str) -> dict:
        return await answer(archs.arch, arch)

    @app.get(f"{PREFIX}/archs/{{arch}}/cost-tree")
    async def cost_tree(arch: str, request: Request) -> dict:
        """One supported parameter set's cost tree. The query names the set:
        ``gpu``, ``model`` (a model config stem) and every param the arch's
        ``#[supported]`` rows name (``/archs/{arch}`` lists them as ``query``).
        A missing, unknown or mistyped param is a 400 and a set no row covers a
        404; both list the valid ``choices``."""

        return await answer(archs.cost_tree, arch, dict(request.query_params))

    return app
