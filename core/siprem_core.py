"""Pipeline matemático principal do SiPREM."""

from __future__ import annotations
import numpy as np

from .data_loader import CRIT_DIR, PREF_TYPES, SCALE_MAX, fuzzify_by_scale
from .promethee import run_fuzzy_promethee_trap, rank_from_net_flow
from .classification import key_by_majority, classify_requirements_kmeans
from .fuzzy_numbers import defuzz_coa


def build_fuzzy_matrix(data_numeric):
    data_numeric = np.asarray(data_numeric, dtype=float)
    n_req, n_crit = data_numeric.shape
    F = np.zeros((n_req, n_crit, 4), dtype=float)
    for i in range(n_req):
        for k in range(n_crit):
            F[i, k] = fuzzify_by_scale(data_numeric[i, k], SCALE_MAX[k])
    return F


def aggregate_fuzzy_flows(phi_plus_fuzzy_exp, phi_minus_fuzzy_exp, phi_net_fuzzy_exp):
    plus = np.mean(phi_plus_fuzzy_exp, axis=2)
    minus = np.mean(phi_minus_fuzzy_exp, axis=2)
    net = np.mean(phi_net_fuzzy_exp, axis=2)

    plus_crisp = np.array([defuzz_coa(x) for x in plus])
    minus_crisp = np.array([defuzz_coa(x) for x in minus])
    net_crisp = np.array([defuzz_coa(x) for x in net])

    return plus, minus, net, plus_crisp, minus_crisp, net_crisp


def _thresholds():
   
q_vals = [0, 0.6, 0.6, 0.6, 0.6]
p_vals = [0, 1.9, 1.9, 1.9, 1.9]
    return q_vals, p_vals


def run_siprem(evaluations, weights_fuzzy_individual, random_state=0):
    expert_names = list(evaluations.keys())
    n_exp = len(expert_names)
    n_req = next(iter(evaluations.values())).shape[0]

    q_vals, p_vals = _thresholds()

    phi_plus_exp = np.zeros((n_req, n_exp))
    phi_minus_exp = np.zeros((n_req, n_exp))
    phi_net_exp = np.zeros((n_req, n_exp))
    phi_plus_fuzzy_exp = np.zeros((n_req, 4, n_exp))
    phi_minus_fuzzy_exp = np.zeros((n_req, 4, n_exp))
    phi_net_fuzzy_exp = np.zeros((n_req, 4, n_exp))

    impact_matrix = np.zeros((n_req, n_exp))

    for e, name in enumerate(expert_names):
        data_e = np.asarray(evaluations[name], dtype=float)
        impact_matrix[:, e] = data_e[:, 0]
        F_e = build_fuzzy_matrix(data_e)
        w_e = weights_fuzzy_individual[:, :, e]

        out = run_fuzzy_promethee_trap(
            F_e, w_e, CRIT_DIR, PREF_TYPES, q_vals, p_vals
        )

        pp, pm, pn, ppf, pmf, pnf = out
        phi_plus_exp[:, e] = pp
        phi_minus_exp[:, e] = pm
        phi_net_exp[:, e] = pn
        phi_plus_fuzzy_exp[:, :, e] = ppf
        phi_minus_fuzzy_exp[:, :, e] = pmf
        phi_net_fuzzy_exp[:, :, e] = pnf

    agg = aggregate_fuzzy_flows(
        phi_plus_fuzzy_exp, phi_minus_fuzzy_exp, phi_net_fuzzy_exp
    )
    plus_fuzzy, minus_fuzzy, net_fuzzy, plus, minus, net = agg

    key_idx = key_by_majority(impact_matrix, SCALE_MAX[0])
    categories, km_labels, centroids, group_names = classify_requirements_kmeans(
        net, key_idx, random_state=random_state
    )

    return {
        "expert_names": expert_names,
        "phi_plus_experts": phi_plus_exp,
        "phi_minus_experts": phi_minus_exp,
        "phi_net_experts": phi_net_exp,
        "phi_plus_fuzzy_experts": phi_plus_fuzzy_exp,
        "phi_minus_fuzzy_experts": phi_minus_fuzzy_exp,
        "phi_net_fuzzy_experts": phi_net_fuzzy_exp,
        "phi_plus_fuzzy": plus_fuzzy,
        "phi_minus_fuzzy": minus_fuzzy,
        "phi_net_fuzzy": net_fuzzy,
        "phi_plus": plus,
        "phi_minus": minus,
        "phi_net": net,
        "rank_experts": np.column_stack([
            rank_from_net_flow(phi_net_exp[:, e]) for e in range(n_exp)
        ]),
        "rank_siprem": rank_from_net_flow(net),
        "impact_matrix": impact_matrix,
        "key_indices": key_idx,
        "categories": categories,
        "kmeans_labels": km_labels,
        "kmeans_centroids": centroids,
        "group_names": group_names,
        "q_vals": q_vals,
        "p_vals": p_vals,
    }
