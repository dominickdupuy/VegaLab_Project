"""The dashboard server: the static site in ``web/out`` plus a read-mostly JSON API.

Lane: Ilan Danial (machinery).

    python -m optspread.server --port 8600
    python -m optspread.server --runs runs --data "$VEGALAB_DATA" --workers 2

Thin on purpose. Every route parses its query, calls one builder in
``optspread.webapi`` and returns the typed model it gets back, so what the
browser sees is exactly what the tested builders produce. The only writes are the
ones the harness already makes: a launched job writes an ordinary run directory
through ``write_run``, and nothing else on disk is touched.

FastAPI and uvicorn are the ``dashboard`` extra. Nothing else in the package
imports this module, so ``make check`` runs without them.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import threading
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from optspread import __version__
from optspread import webapi as w
from optspread import webviews as v
from optspread.attribution import default_root
from optspread.dashboard_data import (
    REPO_ROOT,
    RUNS_ENV,
    RunRow,
    docs_dir,
    run_leak_demo,
    runs_root,
)
from optspread.dashboard_panel import data_root
from optspread.jobs import (
    MAX_PENDING,
    JobManager,
    JobNotCancellableError,
    JobSpec,
    QueueFullError,
)
from optspread.manifest import git_sha
from optspread.orchestrate import default_runs_root
from optspread.webcache import Answers, file_stamp, tree_stamp

WEB_DIR = REPO_ROOT / "web" / "out"
CACHE_ENV = "OPTSPREAD_CACHE"
CACHE_DIR = "webcache"  # under the attribution root: the panel's answers, kept on disk
# Hashed, so a new build has new names: the browser may keep these for good.
IMMUTABLE = "public, max-age=31536000, immutable"


def code_stamp() -> str:
    """A digest of this package's source (path, mtime, size): an answer kept on disk by
    one build is never served by another."""
    return tree_stamp(Path(__file__).parent, skip=("__pycache__",))


@dataclass(frozen=True, slots=True)
class Settings:
    runs: Path
    docs: Path
    data: Path
    web: Path
    readme: Path
    workers: int = 2
    host: str = "127.0.0.1"  # the address served on; its name is an allowed Host header
    max_jobs: int = MAX_PENDING  # jobs queued or running before a launch is a 429
    attribution: Path | None = None  # cached attribution; None is attribution.default_root()
    actionmix: Path | None = None  # action-mix files; None is actionmix.default_root()
    cache: Path | None = None  # answers kept on disk; None keeps them in memory only
    warm: bool = False  # read the runs index and the panel's tables in the background at start

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            # $OPTSPREAD_RUNS, else where the orchestrators write: one default runs root.
            runs=runs_root() if os.environ.get(RUNS_ENV) else default_runs_root(),
            docs=docs_dir(),
            data=data_root(),
            web=Path(os.environ.get("OPTSPREAD_WEB", str(WEB_DIR))),
            readme=REPO_ROOT / "README.md",
            workers=int(os.environ.get("OPTSPREAD_WORKERS", "2")),
            max_jobs=int(os.environ.get("OPTSPREAD_MAX_JOBS", str(MAX_PENDING))),
            cache=Path(os.environ[CACHE_ENV]).expanduser() if os.environ.get(CACHE_ENV) else None,
        )


def _day(value: str, name: str = "date") -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(422, f"{name} must be YYYY-MM-DD, got {value!r}") from exc


LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]", "::1")


def allowed_hosts(host: str) -> list[str]:
    """Host headers the server answers: loopback and the address it was started on.

    Anything else is refused, which is what stops DNS rebinding: a page on another
    site that resolves its own name to 127.0.0.1 still sends its own name as Host.
    """
    names = [*LOCAL_HOSTS, host]
    if ":" in host and not host.startswith("["):  # an IPv6 literal is bracketed in Host
        names.append(f"[{host}]")
    return list(dict.fromkeys(names))


def redactor(named: Sequence[tuple[Path, str]]) -> Callable[[object], object]:
    """Replace each folder in ``named`` with its name in an error's text, longest first.

    An error the browser sees says which root a file is missing from, never where that
    root is on the serving machine: ``<runs root>/grid-1a2b.json``, not a home directory.
    A NaN or infinite float becomes None: an undefined figure is sent as null, as pydantic
    writes it, where the strict JSON encoder of every response would refuse it (a 500).
    """
    pairs: dict[str, str] = {}
    for path, name in named:
        for form in (str(path), str(path.expanduser()), str(path.expanduser().resolve())):
            if len(form) > 1:
                pairs.setdefault(form, name)
    ordered = sorted(pairs.items(), key=lambda kv: len(kv[0]), reverse=True)
    # Where the separator is not "/" (Windows), the rest of a path after a root's name is
    # written with "/", as Linux writes it: <runs root>/sweeps/tiny, not <runs root>\sweeps.
    # It runs up to a character no Windows path holds. Linux is left as it is: there a
    # backslash is part of a name.
    names = sorted({re.escape(name) for name in pairs.values()}, key=len, reverse=True)
    native = (
        re.compile("(" + "|".join(names) + r')((?:\\[^\\<>:"|?*\r\n]*)+)')
        if os.sep != "/" and names
        else None
    )

    def redact(value: object) -> object:
        if isinstance(value, str):
            for form, name in ordered:
                value = value.replace(form, name)
            if native is not None:
                value = native.sub(lambda m: m[1] + m[2].replace("\\", "/"), value)
            return value
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, list | tuple):
            return [redact(v) for v in value]
        if isinstance(value, dict):
            return {k: redact(v) for k, v in value.items()}
        return value

    return redact


def create_app(settings: Settings | None = None, manager: JobManager | None = None) -> FastAPI:
    cfg = settings if settings is not None else Settings.from_env()
    reader = w.PanelReader(cfg.data)
    jobs: dict[str, JobManager] = {}
    tasks = v.Tasks()
    attribution_root = cfg.attribution if cfg.attribution is not None else default_root()
    redact = redactor(
        [
            (cfg.runs, "<runs root>"),
            (cfg.data, "<data root>"),
            (cfg.docs, "<docs>"),
            (cfg.web, "<site>"),
            (attribution_root, "<attribution root>"),
            (REPO_ROOT, "<repo>"),
            (Path.home(), "~"),
        ]
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        jobs["manager"] = (
            manager
            if manager is not None
            else JobManager(cfg.runs, cfg.workers, max_pending=cfg.max_jobs)
        )
        if cfg.warm:
            threading.Thread(target=warm_up, name="optspread-warm", daemon=True).start()
        yield
        jobs["manager"].shutdown()
        tasks.shutdown()

    def warm_up() -> None:
        """What the first visitor would otherwise wait for: the runs index, the panel's
        tables (the surface grid is the slow one) and the latest day's quoted chain, read
        once at start so the first page is as fast as the second. Best effort."""
        steps: list[Callable[[], object]] = [
            lambda: w.catalog(cfg.runs, w.index(cfg.runs)),
            reader.coverage,
            reader.daily,
            reader.surface_grid,
            reader.grid_dates,
            reader.atm_term,
        ]
        for step in steps:
            with contextlib.suppress(Exception):
                step()
        with contextlib.suppress(Exception):
            latest = reader.grid_dates()[-1]
            reader.chain_index(date.fromisoformat(latest), None, [])

    # Untyped without the dashboard extra (CI's plain job), typed with it.
    class Redacted(JSONResponse):  # type: ignore[misc, unused-ignore]
        """Every JSON body names a root, never where it is on the serving machine: a
        payload's ``root``, ``file`` or ``run_dir`` reads ``<runs root>/...``, not a home
        directory, as the errors already did."""

        def render(self, content: Any) -> bytes:
            rendered: bytes = super().render(redact(content))
            return rendered

    app = FastAPI(
        title="optspread dashboard",
        version=__version__,
        lifespan=lifespan,
        default_response_class=Redacted,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts(cfg.host))

    def queue() -> JobManager:
        return jobs["manager"]

    # -- kept answers (optspread.webcache) ----------------------------------- #

    answers = Answers(cfg.cache)
    code = code_stamp()

    def runs_stamp() -> str:
        return tree_stamp(cfg.runs)

    def docs_stamp() -> str:
        return tree_stamp(cfg.docs)

    def panel_stamp() -> tuple[str, object]:
        return tree_stamp(reader.layout.panel), file_stamp([cfg.data / "MANIFEST.json"])

    def answer(
        request: Request,
        route: str,
        deps: object,
        build: Callable[[], object],
        *,
        disk: bool = False,
    ) -> Response:
        """``build()`` rendered as the route would render it, kept while ``deps`` (a stamp
        of the files it reads) stand; 304 to a browser that already holds it."""

        def render() -> bytes:
            return bytes(Redacted(jsonable_encoder(build())).body)

        query = {"q": sorted(request.query_params.multi_items())}
        body, etag = answers.get(route, query, (code, deps), render, disk=disk)
        headers = {"ETag": etag, "Cache-Control": "no-cache"}
        if etag in request.headers.get("if-none-match", ""):
            return Response(status_code=304, headers=headers)
        return Response(body, media_type="application/json", headers=headers)

    @app.exception_handler(w.PanelNotBuiltError)
    async def _not_built(_: object, exc: w.PanelNotBuiltError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "detail": redact(f"panel not built yet: {exc}"),
                "missing": list(reader.layout.missing()),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: object, exc: StarletteHTTPException) -> JSONResponse:
        # every error the browser sees names a root, never an absolute path
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": redact(exc.detail)},
            headers=getattr(exc, "headers", None),
        )

    # -- meta -------------------------------------------------------------- #

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "version": __version__, "git_sha": git_sha(REPO_ROOT)}

    @app.get("/api/overview", response_model=w.OverviewOut)
    def overview(request: Request) -> Response:
        deps = (runs_stamp(), docs_stamp(), file_stamp([cfg.readme]), panel_stamp())
        return answer(
            request,
            "overview",
            deps,
            lambda: w.overview(cfg.readme, cfg.docs, cfg.runs, reader, w.index(cfg.runs)),
            disk=True,  # it reads the panel's coverage: a restarted server answers at once
        )

    # -- runs -------------------------------------------------------------- #

    @app.get("/api/runs", response_model=w.RunListOut)
    def runs(
        request: Request,
        config_hash: list[str] = Query(default=[]),  # noqa: B008
        seed: list[int] = Query(default=[]),  # noqa: B008
        policy: list[str] = Query(default=[]),  # noqa: B008
        premium: list[float] = Query(default=[]),  # noqa: B008
        commit: list[str] = Query(default=[]),  # noqa: B008
        sweep: list[str] = Query(default=[]),  # noqa: B008
        source: list[str] = Query(default=[]),  # noqa: B008
        q: str = "",
    ) -> Response:
        filters = w.RunFilters(
            config_hash=config_hash,
            seed=seed,
            policy=policy,
            premium=premium,
            commit=commit,
            sweep=sweep,
            source=source,
            q=q,
        )
        return answer(
            request, "runs", runs_stamp(), lambda: w.run_list(w.index(cfg.runs), cfg.runs, filters)
        )

    @app.get("/api/runs/detail", response_model=w.RunDetailOut)
    def run_detail(request: Request, key: str) -> Response:
        def build() -> w.RunDetailOut:
            try:
                detail = w.load_detail(cfg.runs, key)
            except KeyError as exc:
                raise HTTPException(404, f"no run at {key!r} under the runs root") from exc
            return w.run_detail(detail, cfg.runs, w.index(cfg.runs))

        return answer(request, "runs/detail", runs_stamp(), build)

    @app.get("/api/runs/compare", response_model=w.CompareOut)
    def run_compare(request: Request, a: str, b: str, group: str = "") -> Response:
        def build() -> w.CompareOut:
            try:
                da, db = w.load_detail(cfg.runs, a), w.load_detail(cfg.runs, b)
            except KeyError as exc:
                raise HTTPException(404, f"no run at {exc.args[0]!r} under the runs root") from exc
            index = w.index(cfg.runs)
            return w.compare(da, db, cfg.runs, index, w.catalog(cfg.runs, index), group)

        return answer(request, "runs/compare", runs_stamp(), build)

    @app.get("/api/runs/catalog", response_model=w.CatalogOut)
    def run_catalog(request: Request) -> Response:
        """The runs root's catalog.json, grouped by question; ``available`` False without one."""
        return answer(
            request, "runs/catalog", runs_stamp(), lambda: w.catalog(cfg.runs, w.index(cfg.runs))
        )

    @app.get("/api/runs/premiums", response_model=w.PremiumsOut)
    def run_premiums(
        request: Request,
        config_hash: list[str] = Query(default=[]),  # noqa: B008
        seed: list[int] = Query(default=[]),  # noqa: B008
        policy: list[str] = Query(default=[]),  # noqa: B008
        premium: list[float] = Query(default=[]),  # noqa: B008
        commit: list[str] = Query(default=[]),  # noqa: B008
        sweep: list[str] = Query(default=[]),  # noqa: B008
        source: list[str] = Query(default=[]),  # noqa: B008
        q: str = "",
    ) -> Response:
        filters = w.RunFilters(
            config_hash=config_hash,
            seed=seed,
            policy=policy,
            premium=premium,
            commit=commit,
            sweep=sweep,
            source=source,
            q=q,
        )
        return answer(
            request,
            "runs/premiums",
            runs_stamp(),
            lambda: w.premium_view(w.index(cfg.runs), cfg.runs, filters),
        )

    # -- jobs -------------------------------------------------------------- #

    @app.get("/api/jobs/options")
    def job_options() -> w.JobOptionsOut:
        return w.job_options()

    @app.post("/api/jobs", status_code=201)
    def submit(spec: dict[str, object]) -> w.JobOut:
        try:
            parsed = JobSpec.model_validate(spec)
            job_id = queue().submit(parsed)
        except ValidationError as exc:
            raise HTTPException(422, [e["msg"] for e in exc.errors()]) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except QueueFullError as exc:  # refused, not queued: the browser says so and waits
            raise HTTPException(429, str(exc)) from exc
        return w.job_out(queue().status(job_id))

    @app.get("/api/jobs")
    def job_list() -> list[w.JobOut]:
        return [w.job_out(s) for s in queue().all()]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> w.JobOut:
        if job_id not in queue().jobs:
            raise HTTPException(404, f"no job {job_id!r}")
        return w.job_out(queue().status(job_id))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> w.JobOut:
        """A queued job never runs; a running one is stopped or its result discarded."""
        if job_id not in queue().jobs:
            raise HTTPException(404, f"no job {job_id!r}")
        try:
            return w.job_out(queue().cancel(job_id))
        except JobNotCancellableError as exc:
            raise HTTPException(409, str(exc)) from exc

    # -- the power lab and the leak demo ----------------------------------- #

    @app.get("/api/power", response_model=w.PowerOut)
    def power(
        request: Request,
        tau2: float = w.POWER_DEFAULTS["tau2"],
        sigma2: float = w.POWER_DEFAULTS["sigma2"],
        n_seeds: int = 10,
        n_episodes: int = 100,
        delta: float = w.POWER_DEFAULTS["delta"],
        alpha: float = 0.05,
        target_power: float = 0.80,
        train_steps: int = 1_000_000,
        episode_days: int = 63,
        train_rate: float = 1_000.0,
        eval_rate: float = 1_800.0,
        cores: int = 1,
    ) -> Response:
        try:
            params = w.PowerParams(
                tau2=tau2,
                sigma2=sigma2,
                n_seeds=n_seeds,
                n_episodes=n_episodes,
                delta=delta,
                alpha=alpha,
                target_power=target_power,
                train_steps=train_steps,
                episode_days=episode_days,
                train_rate=train_rate,
                eval_rate=eval_rate,
                cores=cores,
            )
        except ValidationError as exc:
            raise HTTPException(422, [e["msg"] for e in exc.errors()]) from exc

        def build() -> w.PowerOut:
            try:
                return w.power_lab(params)
            except (ArithmeticError, ValueError) as exc:  # a design the floats cannot hold
                raise HTTPException(422, f"this design cannot be computed: {exc}") from exc

        return answer(request, "power", (), build)

    @app.get("/api/power/surface", response_model=w.PowerSurfaceOut)
    def power_surface(
        request: Request,
        tau2: float = w.POWER_DEFAULTS["tau2"],
        sigma2: float = w.POWER_DEFAULTS["sigma2"],
        n_episodes: int = 100,
        delta: float = w.POWER_DEFAULTS["delta"],
        alpha: float = 0.05,
        target_power: float = 0.80,
        max_premium: float = 4.0,
        max_seeds: int = 100,
    ) -> Response:
        try:
            params = w.PowerSurfaceParams(
                tau2=tau2,
                sigma2=sigma2,
                n_episodes=n_episodes,
                delta=delta,
                alpha=alpha,
                target_power=target_power,
                max_premium=max_premium,
                max_seeds=max_seeds,
            )
        except ValidationError as exc:
            raise HTTPException(422, [e["msg"] for e in exc.errors()]) from exc

        def build() -> w.PowerSurfaceOut:
            try:
                return w.power_surface(params)
            except (ArithmeticError, ValueError) as exc:
                raise HTTPException(422, f"this design cannot be computed: {exc}") from exc

        return answer(request, "power/surface", (), build)

    @app.post("/api/leaks/demo")
    def leaks(params: w.LeakDemoParams) -> w.LeakDemoOut:
        return w.leak_demo(run_leak_demo(eps=params.eps, n_seeds=params.n_seeds))

    # -- sweeps ------------------------------------------------------------ #

    @app.get("/api/sweeps", response_model=w.SweepListOut)
    def sweep_list(request: Request) -> Response:
        return answer(request, "sweeps", runs_stamp(), lambda: w.sweeps(cfg.runs))

    @app.get("/api/sweeps/detail", response_model=w.SweepOut)
    def sweep_detail(request: Request, key: str) -> Response:
        def build() -> w.SweepOut:
            try:
                return w.sweep_detail(cfg.runs, key)
            except KeyError as exc:
                raise HTTPException(404, f"no sweep at {key!r} under the runs root") from exc

        return answer(request, "sweeps/detail", runs_stamp(), build)

    # -- the action mix ---------------------------------------------------- #

    @app.get("/api/runs/actionmix")
    def action_mix(
        run: str | None = None, grid: str | None = None, source: str = "synthetic"
    ) -> w.ActionMixOut:
        try:
            return w.action_mix(cfg.actionmix, run=run, grid=grid, source=source)
        except KeyError as exc:
            raise HTTPException(
                404, f"no action mix for {exc.args[0]!r} yet; make actionmix writes it"
            ) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    # -- the SPX panel ------------------------------------------------------ #
    # Every panel answer is kept on disk too: the panel sits in a synced cloud folder,
    # whose first read of a file can take minutes.

    def from_panel(
        request: Request, route: str, build: Callable[[], object], extra: object = ()
    ) -> Response:
        return answer(request, route, (panel_stamp(), extra), build, disk=True)

    def unprocessable(build: Callable[[], object]) -> Callable[[], object]:
        def run() -> object:
            try:
                return build()
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

        return run

    @app.get("/api/data/status", response_model=w.DataStatusOut)
    def data_status(request: Request) -> Response:
        return answer(request, "data/status", panel_stamp(), reader.status)

    @app.get("/api/data/coverage", response_model=w.CoverageOut)
    def coverage(request: Request) -> Response:
        return from_panel(request, "data/coverage", reader.coverage_out)

    @app.get("/api/data/dates", response_model=list[str])
    def dates(request: Request, source: str | None = None) -> Response:
        return from_panel(request, "data/dates", lambda: reader.dates(source))

    @app.get("/api/data/chain", response_model=w.ChainIndexOut)
    def chain(
        request: Request,
        date: str,
        source: str | None = None,
        roots: list[str] = Query(default=[]),  # noqa: B008
    ) -> Response:
        day = _day(date)
        return from_panel(request, "data/chain", lambda: reader.chain_index(day, source, roots))

    @app.get("/api/data/smile", response_model=w.SmileOut)
    def smile(
        request: Request,
        date: str,
        expiry: str | None = None,
        source: str | None = None,
        roots: list[str] = Query(default=[]),  # noqa: B008
    ) -> Response:
        day, exp = _day(date), _day(expiry, "expiry") if expiry else None
        return from_panel(request, "data/smile", lambda: reader.smile_out(day, exp, source, roots))

    @app.get("/api/data/surface", response_model=w.SurfaceOut)
    def surface(
        request: Request,
        date: str,
        source: str | None = None,
        roots: list[str] = Query(default=[]),  # noqa: B008
        min_dte: int = 1,
        max_dte: int = 365,
    ) -> Response:
        day = _day(date)
        return from_panel(
            request,
            "data/surface",
            unprocessable(lambda: reader.surface_out(day, source, roots, min_dte, max_dte)),
        )

    @app.get("/api/data/term", response_model=w.TermOut)
    def term(
        request: Request,
        date: str,
        source: str | None = None,
        roots: list[str] = Query(default=[]),  # noqa: B008
    ) -> Response:
        day = _day(date)
        return from_panel(request, "data/term", lambda: reader.term_out(day, source, roots))

    @app.get("/api/data/daily", response_model=w.DailyOut)
    def daily(request: Request, start: str | None = None, end: str | None = None) -> Response:
        lo = _day(start, "start") if start else None
        hi = _day(end, "end") if end else None
        return from_panel(request, "data/daily", lambda: reader.daily_out(lo, hi))

    @app.get("/api/data/seam", response_model=w.SeamOut)
    def seam(request: Request) -> Response:
        return from_panel(request, "data/seam", reader.seam_out)

    # -- the 3D views over the panel ------------------------------------------ #

    @app.get("/api/data/grid/dates", response_model=list[str])
    def grid_dates(request: Request) -> Response:
        return from_panel(request, "data/grid/dates", reader.grid_dates)

    @app.get("/api/data/grid", response_model=w.GridOut)
    def grid(request: Request, date: str) -> Response:
        day = _day(date)
        return from_panel(request, "data/grid", lambda: reader.grid_out(day))

    @app.get("/api/data/grid/frames", response_model=w.GridFramesOut)
    def grid_frames(
        request: Request, start: str, end: str | None = None, max_frames: int = 126
    ) -> Response:
        lo, hi = _day(start, "start"), _day(end, "end") if end else None
        return from_panel(
            request,
            "data/grid/frames",
            unprocessable(lambda: reader.grid_frames(lo, hi, max_frames)),
        )

    @app.get("/api/data/simulated", response_model=w.SimulatedOut)
    def simulated(
        request: Request, date: str, premium: float = 0.0, sigma_from: str = "config"
    ) -> Response:
        if not (math.isfinite(premium) and 0.0 <= premium <= 20.0):
            raise HTTPException(422, f"premium must be between 0 and 20 vol points, got {premium}")
        day = _day(date)
        return from_panel(
            request,
            "data/simulated",
            unprocessable(lambda: reader.simulated_out(day, premium, sigma_from)),
        )

    # -- the data page's questions ---------------------------------------------- #

    @app.get("/api/data/premium", response_model=w.PremiumOut)
    def premium(request: Request) -> Response:
        return from_panel(
            request,
            "data/premium",
            lambda: reader.premium_out(v.vrp_mapping(cfg.docs).mapping),
            docs_stamp(),
        )

    @app.get("/api/data/shape", response_model=w.ShapeOut)
    def shape(request: Request, date: str) -> Response:
        day = _day(date)
        return from_panel(request, "data/shape", lambda: reader.shape_out(day))

    @app.get("/api/data/quality", response_model=w.QualityOut)
    def quality(request: Request) -> Response:
        return from_panel(
            request, "data/quality", lambda: reader.quality_out(cfg.docs), docs_stamp()
        )

    @app.get("/api/data/landscape", response_model=w.LandscapeOut)
    def landscape(
        request: Request, start: str | None = None, end: str | None = None, max_dates: int = 420
    ) -> Response:
        lo = _day(start, "start") if start else None
        hi = _day(end, "end") if end else None
        return from_panel(
            request,
            "data/landscape",
            unprocessable(lambda: reader.landscape_out(lo, hi, max_dates)),
        )

    # -- the study, its checks and its analyses ------------------------------ #

    @app.get("/api/study", response_model=v.StudyOut)
    def study(request: Request) -> Response:
        return answer(request, "study", docs_stamp(), lambda: v.study(cfg.docs))

    @app.get("/api/checks", response_model=list[v.CheckOut])
    def checks(request: Request) -> Response:
        return answer(request, "checks", docs_stamp(), lambda: v.checks(cfg.docs))

    @app.get("/api/docs/{name}")
    def doc(name: str) -> FileResponse:
        try:
            path = v.doc_file(cfg.docs, name)
        except KeyError as exc:
            raise HTTPException(404, f"no {name!r} under docs/") from exc
        kind = "application/json" if path.suffix == ".json" else "text/plain; charset=utf-8"
        return FileResponse(path, media_type=kind)

    @app.get("/api/vrp", response_model=v.VrpOut)
    def vrp(request: Request) -> Response:
        return answer(request, "vrp", docs_stamp(), lambda: v.vrp_mapping(cfg.docs))

    @app.get("/api/m4", response_model=v.M4Out)
    def m4(request: Request) -> Response:
        return answer(request, "m4", docs_stamp(), lambda: v.m4_analysis(cfg.docs))

    # -- the edge of a grid, by component and structure (optspread.edge) ------- #

    def edge_stamp() -> tuple[str, str]:
        return runs_stamp(), tree_stamp(attribution_root, skip=(CACHE_DIR,))

    @app.get("/api/edge", response_model=v.EdgeListOut)
    def edge_list(request: Request) -> Response:
        return answer(
            request, "edge", edge_stamp(), lambda: v.edge_list(attribution_root, cfg.runs)
        )

    @app.get("/api/edge/detail", response_model=v.EdgeOut)
    def edge_detail(request: Request, grid: str, first: int | None = None) -> Response:
        def build() -> v.EdgeOut:
            try:
                return v.edge_detail(attribution_root, cfg.runs, grid, first)
            except KeyError as exc:
                raise HTTPException(
                    404, f"no edge for grid {exc.args[0]!r} yet; make edge writes it"
                ) from exc
            except ValueError as exc:
                raise HTTPException(422, str(exc).splitlines()[0]) from exc

        return answer(request, "edge/detail", edge_stamp(), build)

    # -- one run: attribution, capital, replay -------------------------------- #

    def run_row(key: str) -> tuple[Path, RunRow]:
        try:
            detail = w.load_detail(cfg.runs, key)
        except KeyError as exc:
            raise HTTPException(404, f"no run at {key!r} under the runs root") from exc
        return detail.row.path, detail.row

    @app.get("/api/runs/attribution")
    def attribution(key: str, seed: int | None = None) -> v.AttributionOut:
        _, row = run_row(key)
        state, _ = tasks.state("attribution", key)
        return v.attribution_view(attribution_root, row, state, seed)

    @app.post("/api/runs/attribution", status_code=202)
    def attribute(key: str) -> v.AttributionOut:
        run_dir, row = run_row(key)
        reason = v.attribution_reason(row)
        if reason:
            raise HTTPException(422, reason)
        target = v.attribution_file(attribution_root, row)
        if not target.is_file():
            tasks.submit("attribution", key, partial(v.compute_attribution, run_dir, target))
        state, _ = tasks.state("attribution", key)
        return v.attribution_view(attribution_root, row, state)

    @app.get("/api/runs/capital")
    def capital(key: str) -> v.CapitalOut:
        _, row = run_row(key)
        return v.capital_view(row, *tasks.state("capital", key), attribution_root)

    @app.post("/api/runs/capital", status_code=202)
    def capital_start(key: str) -> v.CapitalOut:
        run_dir, row = run_row(key)
        reason = v.capital_reason(row)
        if reason:
            raise HTTPException(422, reason)
        target = v.capital_file(attribution_root, row)
        if v.read_capital(target) is None:
            tasks.submit("capital", key, partial(v.compute_capital, run_dir, target))
        return v.capital_view(row, *tasks.state("capital", key), attribution_root)

    @app.get("/api/runs/mixview")
    def mix_view(key: str) -> v.MixViewOut:
        _, row = run_row(key)
        state, _ = tasks.state("mix", key)
        return v.mix_view(attribution_root, row, state, cfg.actionmix)

    @app.post("/api/runs/mixview", status_code=202)
    def mix_start(key: str) -> v.MixViewOut:
        run_dir, row = run_row(key)
        reason = v.mix_reason(row)
        if reason:
            raise HTTPException(422, reason)
        target = v.mix_file(attribution_root, row)
        if not target.is_file():
            tasks.submit("mix", key, partial(v.compute_mix, run_dir, target))
        state, _ = tasks.state("mix", key)
        return v.mix_view(attribution_root, row, state, cfg.actionmix)

    @app.get("/api/runs/replay")
    def replay(key: str) -> v.ReplayOut:
        _, row = run_row(key)
        return v.replay_view(row, *tasks.state("replay", key))

    @app.post("/api/runs/replay", status_code=202)
    def replay_start(key: str) -> v.ReplayOut:
        run_dir, row = run_row(key)
        reason = v.replay_reason(row)
        if reason:
            raise HTTPException(422, reason)
        tasks.submit("replay", key, partial(v.run_replay, run_dir))
        return v.replay_view(row, *tasks.state("replay", key))

    # -- the leak detector's floor and fields --------------------------------- #

    @app.get("/api/leaks/fields")
    def leak_fields() -> list[str]:
        return v.leak_fields()

    @app.post("/api/leaks/floor")
    def leak_floor(params: v.LeakFloorParams) -> v.LeakFloorOut:
        try:
            return v.leak_floor_out(params)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/leaks/field")
    def leak_field(params: v.LeakFieldParams) -> v.LeakFieldOut:
        try:
            return v.leak_field_out(params)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/leaks/coverage")
    def leak_coverage(params: v.LeakFieldParams) -> v.LeakCoverageOut:
        return v.leak_coverage_out(params.eps, params.n_seeds)

    # -- the seller on a real close, and launch ------------------------------- #

    @app.get("/api/data/seller", response_model=v.SellerOut)
    def seller(request: Request, date: str) -> Response:
        day = _day(date)
        return from_panel(request, "data/seller", lambda: v.seller_out(reader, day))

    @app.get("/api/launch/options")
    def launch_options() -> w.JobOptionsOut:
        return v.launch_options()

    @app.get("/api/launch/estimate")
    def launch_estimate(
        n_episodes: int, n_days: int, total_timesteps: int | None = None
    ) -> v.EstimateOut:
        if n_episodes < 1 or n_days < 1 or (total_timesteps is not None and total_timesteps < 0):
            raise HTTPException(422, "episodes and days must be positive, steps not negative")
        return v.launch_estimate(n_episodes, n_days, total_timesteps)

    @app.post("/api/launch/preview")
    def launch_preview(spec: dict[str, object]) -> v.PreviewOut:
        """What a spec would run, or every reason it would be refused: never a 422, so the
        form can show the reasons as the knobs change."""
        return v.launch_preview(spec, cfg.runs, w.index(cfg.runs))

    # An /api path no route serves is a JSON 404, not the site's HTML page; a route asked
    # with a method it does not take is still a 405 that names the ones it does.
    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    def no_route(rest: str, request: Request) -> None:
        allowed = sorted(
            {
                m
                for r in app.routes
                if isinstance(r, APIRoute)
                and r.name != "no_route"
                and r.path_regex.match(request.url.path)
                for m in r.methods or ()
            }
        )
        if allowed:
            raise HTTPException(405, "method not allowed", headers={"Allow": ", ".join(allowed)})
        raise HTTPException(404, f"no API route /api/{rest}")

    # -- the site ------------------------------------------------------------ #

    @app.middleware("http")
    async def immutable_assets(request: Request, call_next: Callable[[Request], Any]) -> Any:
        response = await call_next(request)
        if request.url.path.startswith("/_next/static/") and response.status_code == 200:
            response.headers["Cache-Control"] = IMMUTABLE
        return response

    if cfg.web.is_dir():
        app.mount("/", StaticFiles(directory=cfg.web, html=True), name="web")
    else:

        @app.get("/")
        def no_site() -> dict[str, str]:
            return {"detail": "no built site; run `make web`, or `make web-dev`"}

    return app


def main() -> int:  # pragma: no cover - exercised by `make serve`, not by the test suite
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the optspread dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--runs", type=Path, default=None, help="runs root (default: runs/)")
    parser.add_argument("--docs", type=Path, default=None, help="gate reports (default: docs/)")
    parser.add_argument("--data", type=Path, default=None, help="data root ($VEGALAB_DATA)")
    parser.add_argument("--web", type=Path, default=None, help="built site (default: web/out)")
    parser.add_argument("--workers", type=int, default=2, help="job worker processes")
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help=f"answers kept on disk (default: ${CACHE_ENV}, else <attribution root>/{CACHE_DIR})",
    )
    args = parser.parse_args()
    base = Settings.from_env()
    cache = args.cache or base.cache or default_root() / CACHE_DIR
    settings = Settings(
        runs=(args.runs or base.runs).resolve(),
        docs=(args.docs or base.docs).resolve(),
        data=(args.data or base.data).expanduser(),
        web=(args.web or base.web).resolve(),
        readme=base.readme,
        workers=int(args.workers),
        host=str(args.host),
        cache=cache.expanduser().resolve(),
        warm=True,
    )
    print(f"runs {settings.runs}\ndocs {settings.docs}\ndata {settings.data}\nweb  {settings.web}")
    uvicorn.run(create_app(settings), host=args.host, port=int(args.port))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
