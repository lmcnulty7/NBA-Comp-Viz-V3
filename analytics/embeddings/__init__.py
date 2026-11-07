"""analytics.embeddings: player style embeddings, unit player-season, unsupervised (docs/contract/embeddings.md).

Modules: validate (input checks), features (player_season_features), train (PCA and the gated autoencoder, the
evaluation and the report), eda (tables for notebooks/eda_embeddings.ipynb). Held-out games of sportvu/splits.json
never enter any aggregate; see analytics/embeddings/train.py for the protocol.
"""
