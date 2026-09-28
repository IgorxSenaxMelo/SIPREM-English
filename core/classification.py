"""Classificação do SiPREM conforme classify_requirements_kmeans do MATLAB."""

from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans


def classify_requirements_kmeans(phi_vec, key_idx, random_state=0):
    """Reproduz a lógica MATLAB.

    O MATLAB usa Replicates=20, Start='plus' e Distance='sqeuclidean'.
    random_state é explicitado no Python para permitir reprodutibilidade.
    """
    phi_vec = np.asarray(phi_vec, dtype=float).reshape(-1)
    n_req = len(phi_vec)

    cat = np.full(n_req, "", dtype=object)
    key_idx = np.unique(np.asarray(key_idx, dtype=int))

    if key_idx.size and np.any(key_idx >= n_req):
        raise IndexError("key_idx contém índice fora do vetor.")

    if key_idx.size:
        cat[key_idx] = "Key"

    key_set = set(key_idx.tolist())
    rest = np.array([i for i in range(n_req) if i not in key_set], dtype=int)
    n_rest = len(rest)

    if n_rest == 0:
        return cat, np.full(n_req, np.nan), np.array([]), []

    if n_rest == 1:
        cat[rest[0]] = "Grupo 1"
        return cat, np.full(n_req, np.nan), np.array([]), ["Grupo 1"]

    if n_rest == 2:
        order = rest[np.argsort(phi_vec[rest])[::-1]]
        cat[order[0]] = "Grupo 1"
        cat[order[1]] = "Grupo 2"
        return cat, np.full(n_req, np.nan), np.array([]), ["Grupo 1", "Grupo 2"]

    k = min(3, n_rest)
    model = KMeans(
        n_clusters=k,
        n_init=20,
        init="k-means++",
        max_iter=300,
        random_state=random_state,
        algorithm="lloyd",
    )

    labels = model.fit_predict(phi_vec[rest].reshape(-1, 1))
    centers = model.cluster_centers_.reshape(-1)
    ord_c = np.argsort(centers)[::-1]

    cluster_to_group = {
        int(cluster): f"Grupo {position}"
        for position, cluster in enumerate(ord_c, start=1)
    }

    for idx, cluster in zip(rest, labels):
        cat[idx] = cluster_to_group[int(cluster)]

    idx_full = np.full(n_req, np.nan)
    idx_full[rest] = labels

    return cat, idx_full, centers[ord_c], [f"Grupo {i}" for i in range(1, k+1)]


def key_by_majority(impacto_individual, scale_max=7):
    """KEY: pelo menos 50% dos especialistas deram impacto máximo."""
    impacto = np.asarray(impacto_individual, dtype=float)
    return np.where(
        np.sum(impacto == scale_max, axis=1) / impacto.shape[1] >= 0.5
    )[0]
