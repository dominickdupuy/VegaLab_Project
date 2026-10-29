"""optspread: a leak-tested SPX options simulator used to measure a detection floor.

Lanes (see README.md, The lane map):
    market layer      -- Hemdutt Rao    (rng and types contracts, pricing, market,
                                         effects, costs, margin, actions, portfolio,
                                         policies, scripted, data, extend, realdata,
                                         vrp)
    machinery         -- Ilan Danial    (config, manifest, env, harness, train, sweep,
                                         orchestrate, replay, cli, jobs, the dashboard,
                                         CI)
    statistical spine -- Dominick Dupuy (leaks, power, analysis, validation,
                                         attribution, edge, actionmix, m4analysis,
                                         the gate reports, reproduction)
"""

__version__ = "0.1.0"
