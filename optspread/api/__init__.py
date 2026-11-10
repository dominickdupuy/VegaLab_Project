"""The dashboard's API: typed payloads built from what the harness wrote, one module per
area of the site.

Lane: Ilan Danial (machinery).

``_base`` holds the shared base model and the JSON-safe conversions; every other module
is one area (runs, compare, catalog, premiums, gates, jobs, power, leaks, sweeps, mix,
panel, readings, overview, study, runwork, seller, launch, edge). Nothing is imported
here, so importing one area imports neither the others nor the web framework.
"""
