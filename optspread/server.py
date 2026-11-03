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
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from optspread import __version__
from optspread import webapi as w
from optspread.dashboard_data import REPO_ROOT, docs_dir, run_leak_demo, runs_root
from optspread.dashboard_panel import data_root
from optspread.jobs import JobManager, JobSpec
from optspread.manifest import git_sha

WEB_DIR = REPO_ROOT / "web" / "out"


@dataclass(frozen=True, slots=True)
class Settings:
    runs: Path
    docs: Path
    data: Path
    web: Path
    readme: Path
    workers: int = 2
    host: str = "127.0.0.1"  # the address served on; its name is an allowed Host header

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            runs=runs_root(),
            docs=docs_dir(),
            data=data_root(),
            web=Path(os.environ.get("OPTSPREAD_WEB", str(WEB_DIR))),
            readme=REPO_ROOT / "README.md",
            workers=int(os.environ.get("OPTSPREAD_WORKERS", "2")),
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


def create_app(settings: Settings | None = None, manager: JobManager | None = None) -> FastAPI:
    cfg = settings if settings is not None else Settings.from_env()
    reader = w.PanelReader(cfg.data)
    jobs: dict[str, JobManager] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        jobs["manager"] = manager if manager is not None else JobManager(cfg.runs, cfg.workers)
        yield
        jobs["manager"].shutdown()

    app = FastAPI(title="optspread dashboard", version=__version__, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts(cfg.host))

    def queue() -> JobManager:
        return jobs["manager"]

    @app.exception_handler(w.PanelNotBuiltError)
    async def _not_built(_: object, exc: w.PanelNotBuiltError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"panel not built yet: {exc}",
                "missing": list(reader.layout.missing()),
            },
        )

    # -- meta -------------------------------------------------------------- #

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "version": __version__, "git_sha": git_sha(REPO_ROOT)}

    @app.get("/api/overview")
    def overview() -> w.OverviewOut:
        return w.overview(cfg.readme, cfg.docs, cfg.runs, reader, w.index(cfg.runs))

    # -- runs -------------------------------------------------------------- #

    @app.get("/api/runs")
    def runs(
        config_hash: list[str] = Query(default=[]),  # noqa: B008
        seed: list[int] = Query(default=[]),  # noqa: B008
        policy: list[str] = Query(default=[]),  # noqa: B008
        premium: list[float] = Query(default=[]),  # noqa: B008
        commit: list[str] = Query(default=[]),  # noqa: B008
        sweep: list[str] = Query(default=[]),  # noqa: B008
        source: list[str] = Query(default=[]),  # noqa: B008
        q: str = "",
    ) -> w.RunListOut:
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
        return w.run_list(w.index(cfg.runs), cfg.runs, filters)

    @app.get("/api/runs/detail")
    def run_detail(key: str) -> w.RunDetailOut:
        try:
            detail = w.load_detail(cfg.runs, key)
        except KeyError as exc:
            raise HTTPException(404, f"no run at {key!r} under {cfg.runs}") from exc
        return w.run_detail(detail, cfg.runs, w.index(cfg.runs))

    @app.get("/api/runs/compare")
    def run_compare(a: str, b: str) -> w.CompareOut:
        try:
            da, db = w.load_detail(cfg.runs, a), w.load_detail(cfg.runs, b)
        except KeyError as exc:
            raise HTTPException(404, f"no run at {exc.args[0]!r} under {cfg.runs}") from exc
        return w.compare(da, db, cfg.runs, w.index(cfg.runs))

    @app.get("/api/runs/policy")
    def run_policy_heatmap(key: str) -> w.PolicyHeatmapOut:
        try:
            run_dir = w.resolve_run(cfg.runs, key)
        except KeyError as exc:
            raise HTTPException(404, f"no run at {key!r} under {cfg.runs}") from exc
        result = w.policy_heatmap(run_dir)
        if result is None:
            raise HTTPException(404, f"no action_log.json for run {key!r}")
        return result

    # -- gates ------------------------------------------------------------- #

    @app.get("/api/gates")
    def gates() -> w.GatesOut:
        return w.gates(cfg.docs)

    @app.get("/api/gates/{milestone}")
    def gate(milestone: str) -> w.GateBoardOut:
        for board in w.gates(cfg.docs).boards:
            if board.milestone.lower() == milestone.lower():
                return board
        raise HTTPException(404, f"no {milestone} gate report under {cfg.docs}")

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
        return w.job_out(queue().status(job_id))

    @app.get("/api/jobs")
    def job_list() -> list[w.JobOut]:
        return [w.job_out(s) for s in queue().all()]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> w.JobOut:
        if job_id not in queue().jobs:
            raise HTTPException(404, f"no job {job_id!r}")
        return w.job_out(queue().status(job_id))

    # -- the power lab and the leak demo ----------------------------------- #

    @app.get("/api/power")
    def power(
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
    ) -> w.PowerOut:
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
        try:
            return w.power_lab(params)
        except (ArithmeticError, ValueError) as exc:  # a design the floats cannot hold
            raise HTTPException(422, f"this design cannot be computed: {exc}") from exc

    @app.post("/api/leaks/demo")
    def leaks(params: w.LeakDemoParams) -> w.LeakDemoOut:
        return w.leak_demo(run_leak_demo(eps=params.eps, n_seeds=params.n_seeds))

    # -- synthetic data preview -------------------------------------------- #

    @app.get("/api/synth")
    def synth(
        sigma: float = 0.20,
        vol_of_vol: float = 0.0,
        skew_slope: float = 0.0,
        alpha_bps: float = 0.0,
        n_days: int = 63,
        n_paths: int = 5,
        seed: int = 0,
    ) -> w.SynthPathsOut:
        from optspread.config import MarketConfig
        try:
            cfg = MarketConfig(
                sigma=sigma,
                vol_of_vol=vol_of_vol,
                skew_slope=skew_slope,
                alpha_bps=alpha_bps,
                n_days=n_days,
            )
        except Exception as exc:
            raise HTTPException(422, str(exc)) from exc
        return w.synth_paths(cfg, n_paths=n_paths, seed=seed)

    # -- sweeps ------------------------------------------------------------ #

    @app.get("/api/sweeps")
    def sweep_list() -> w.SweepListOut:
        return w.sweeps(cfg.runs)

    @app.get("/api/sweeps/detail")
    def sweep_detail(key: str) -> w.SweepOut:
        try:
            return w.sweep_detail(cfg.runs, key)
        except KeyError as exc:
            raise HTTPException(404, f"no sweep at {key!r} under {cfg.runs}") from exc

    # -- the SPX panel ------------------------------------------------------ #

    @app.get("/api/data/status")
    def data_status() -> w.DataStatusOut:
        return reader.status()

    @app.get("/api/data/coverage")
    def coverage() -> w.CoverageOut:
        return reader.coverage_out()

    @app.get("/api/data/dates")
    def dates(source: str | None = None) -> list[str]:
        return reader.dates(source)

    @app.get("/api/data/chain")
    def chain(
        date: str,
        source: str | None = None,
        roots: list[str] = Query(default=[]),  # noqa: B008
    ) -> w.ChainIndexOut:
        return reader.chain_index(_day(date), source, roots)

    @app.get("/api/data/smile")
    def smile(
        date: str,
        expiry: str | None = None,
        source: str | None = None,
        roots: list[str] = Query(default=[]),  # noqa: B008
    ) -> w.SmileOut:
        exp = _day(expiry, "expiry") if expiry else None
        return reader.smile_out(_day(date), exp, source, roots)

    @app.get("/api/data/surface")
    def surface(
        date: str,
        source: str | None = None,
        roots: list[str] = Query(default=[]),  # noqa: B008
        min_dte: int = 1,
        max_dte: int = 365,
    ) -> w.SurfaceOut:
        try:
            return reader.surface_out(_day(date), source, roots, min_dte, max_dte)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/data/term")
    def term(
        date: str,
        source: str | None = None,
        roots: list[str] = Query(default=[]),  # noqa: B008
    ) -> w.TermOut:
        return reader.term_out(_day(date), source, roots)

    @app.get("/api/data/daily")
    def daily(start: str | None = None, end: str | None = None) -> w.DailyOut:
        return reader.daily_out(
            _day(start, "start") if start else None, _day(end, "end") if end else None
        )

    @app.get("/api/data/seam")
    def seam() -> w.SeamOut:
        return reader.seam_out()

    # -- the site ------------------------------------------------------------ #

    if cfg.web.is_dir():
        app.mount("/", StaticFiles(directory=cfg.web, html=True), name="web")
    else:

        @app.get("/")
        def no_site() -> dict[str, str]:
            return {"detail": f"no built site at {cfg.web}; run `make web`, or `make web-dev`"}

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
    args = parser.parse_args()
    base = Settings.from_env()
    settings = Settings(
        runs=(args.runs or base.runs).resolve(),
        docs=(args.docs or base.docs).resolve(),
        data=(args.data or base.data).expanduser(),
        web=(args.web or base.web).resolve(),
        readme=base.readme,
        workers=int(args.workers),
        host=str(args.host),
    )
    print(f"runs {settings.runs}\ndocs {settings.docs}\ndata {settings.data}\nweb  {settings.web}")
    uvicorn.run(create_app(settings), host=args.host, port=int(args.port))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
