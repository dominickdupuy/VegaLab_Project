"""The dashboard's API: typed payloads built from what the harness wrote, and the server.

Lane: Ilan Danial (machinery).

``_base`` holds the shared base model and the JSON-safe conversions; every other payload
module is one area of the site (runs, compare, catalog, premiums, gates, jobs, power,
leaks, sweeps, mix, panel, readings, overview, study, runwork, seller, launch, edge).
``builders`` gathers what ``server`` calls, ``server`` is the FastAPI app and ``cache``
its answer cache. Nothing is imported here, so importing one area imports neither the
others nor the web framework.
"""
