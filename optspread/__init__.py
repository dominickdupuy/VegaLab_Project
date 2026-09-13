"""optspread: a leak-tested SPX options simulator used to measure a detection floor.

Lanes (see README.md):
    market layer      -- Hemdutt Rao    (pricing, market, costs, margin, actions,
                                         portfolio, policies)
    machinery         -- Ilan Danial    (config, manifest, env, harness, CI, bench)
    statistical spine -- Dominick Dupuy (types/rng contracts, leaks, later: power analysis)
"""

__version__ = "0.1.0"
