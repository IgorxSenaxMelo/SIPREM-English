"""Leitura das planilhas reais do SiPREM."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .fahp import group_weights_from_reference_rows, saaty_linguistic2num


CRITERIA = [
    "Impacto",
    "Custo",
    "CAPACIDADE NACIONAL DE IMPLEMENTAÇÃO",
    "Prazo",
    "RiscoVerif",
]

CRIT_DIR = np.array([1, -1, 1, -1, -1], dtype=int)
PREF_TYPES = np.array([1, 4, 4, 4, 4], dtype=int)
SCALE_MAX = np.array([7, 7, 7, 7, 7], dtype=int)
SCALE_MIN = np.array([1, 1, 1, 1, 1], dtype=int)




TFN_eval_7 = [
     0   0   1   2
     1   2   2   3
     2   3   4   5
     4   5   5   6
     5   6   7   8
     7   8   8   9
     8   9  10  10
];




def linguistic2num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        raise ValueError("Valor vazio ou missing encontrado na planilha.")

    if isinstance(x, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(x)):
            raise ValueError("Valor NaN/inf encontrado na planilha.")
        return float(x)

    s = str(x).strip().lower()
    mapping = {
        # Portuguese (legacy/current model files)
        "muito alto": 7,
        "alto": 6,
        "moderadamente alto": 5,
        "médio": 4,
        "medio": 4,
        "moderadamente baixo": 3,
        "baixo": 2,
        "muito baixo": 1,
        # English template / web version
        "very high": 7,
        "high": 6,
        "moderately high": 5,
        "medium": 4,
        "moderately low": 3,
        "low": 2,
        "very low": 1,
    }
    if s not in mapping:
        raise ValueError(f"Valor linguístico desconhecido: {x!r}")
    return mapping[s]


def fuzzify_by_scale(val, scale_max):
    v = int(np.rint(float(val)))
    v = max(1, min(int(scale_max), v))

    if scale_max == 5:
        return TFN_EVAL_5[v-1].copy()
    if scale_max == 7:
        return TFN_EVAL_7[v-1].copy()
    if scale_max == 9:
        return TFN_EVAL_9[v-1].copy()
    raise ValueError("Escala não suportada.")


def read_expert_workbook(path):
    path = Path(path)
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names
    expert_sheets = sheets[:-1]

    if not expert_sheets:
        raise ValueError("Nenhuma aba de especialista foi encontrada.")

    evaluations = {}
    ids = None
    descriptions = None

    for sheet in expert_sheets:
        df = pd.read_excel(path, sheet_name=sheet, header=0)

        if df.shape[1] < 7:
            raise ValueError(f"A aba {sheet!r} precisa ter pelo menos 7 colunas.")

        current_ids = df.iloc[:, 0].tolist()
        current_desc = df.iloc[:, 1].tolist()

        if ids is None:
            ids = current_ids
            descriptions = current_desc
        elif current_ids != ids:
            raise ValueError(f"Aba {sheet!r} possui IDs diferentes dos demais especialistas.")

        raw = df.iloc[:, 2:7]
        numeric = np.empty(raw.shape, dtype=float)
        for i in range(raw.shape[0]):
            for k in range(raw.shape[1]):
                numeric[i, k] = linguistic2num(raw.iat[i, k])
        evaluations[sheet] = numeric

    return {
        "sheets": expert_sheets,
        "ids": ids,
        "descriptions": descriptions,
        "evaluations": evaluations,
        "n_experts": len(expert_sheets),
        "n_requirements": len(ids),
        "n_criteria": 5,
    }


def read_fahp_workbook(path):
    path = Path(path)
    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names
    expert_sheets = sheets[:-1]

    rows = []
    for sheet in expert_sheets:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        row = df.iloc[3, 1:6].tolist()  # B4:F4
        rows.append(row)

    result = group_weights_from_reference_rows(rows)
    result["sheets"] = expert_sheets
    return result
