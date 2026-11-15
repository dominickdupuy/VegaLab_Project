"""Synthetic worlds with a planted edge: generate, freeze, replay, benchmark.

``generators`` plants the edge; ``dataset`` freezes paths to disk under a content hash;
``provider`` replays them through the ordinary environment; ``benchmark`` trains an arm on
the frozen train paths and scores it against an oracle, the null twin and a learner
trained on noise; ``jobs`` runs that from the dashboard.
"""
