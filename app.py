from pathlib import Path
import io
import base64
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.data_loader import read_expert_workbook, read_fahp_workbook, CRITERIA, CRIT_DIR, PREF_TYPES, SCALE_MAX
from core.siprem_core import run_siprem, build_fuzzy_matrix
from core.promethee import pref_trap
from core.fuzzy_numbers import trap_add, trap_sub, trap_mul_geldermann, trap_scalar_div, defuzz_coa

# ---------------------------------------------------------------------------
# English display labels
# The mathematical core remains unchanged. These mappings affect presentation
# only; they do not modify the underlying SiPREM calculations.
# ---------------------------------------------------------------------------
CRITERIA_EN = {
    "Impacto": "Mission Impact",
    "Custo": "Cost",
    "CAPACIDADE NACIONAL DE IMPLEMENTAÇÃO": "National Implementation Capability",
    "Prazo": "Schedule",
    "RiscoVerif": "Verification Risk",
}
CATEGORY_EN = {
    "Key": "KEY",
    "Grupo 1": "Group 1",
    "Grupo 2": "Group 2",
    "Grupo 3": "Group 3",
}

def criteria_en():
    return [CRITERIA_EN.get(c, c) for c in CRITERIA]

def categories_en(values):
    return np.array([CATEGORY_EN.get(str(v), str(v)) for v in values], dtype=object)

def load_user_guide():
    guide_path = ROOT / "SiPREM_User_Guide_EN.pdf"
    if not guide_path.exists():
        return None
    return guide_path.read_bytes()

def load_template(filename):
    path = ROOT / filename
    if not path.exists():
        return None
    return path.read_bytes()


def show_template_downloads():
    st.subheader("📥 Input Templates")
    st.caption("Download the two Excel templates, complete them according to the User Guide, and then upload them below.")

    col1, col2 = st.columns(2)
    req = load_template("SiPREM_Requirement_Assessment_Template_EN.xlsx")
    fahp = load_template("SiPREM_FAHP_Criteria_Weights_Template_EN.xlsx")

    with col1:
        if req is not None:
            st.download_button(
                "⬇️ Requirement Assessment Template",
                data=req,
                file_name="SiPREM_Requirement_Assessment_Template_EN.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.error("Requirement Assessment template not found.")

    with col2:
        if fahp is not None:
            st.download_button(
                "⬇️ FAHP-Express Criteria Weights Template",
                data=fahp,
                file_name="SiPREM_FAHP_Criteria_Weights_Template_EN.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.error("FAHP-Express template not found.")


def show_user_guide():
    pdf_bytes = load_user_guide()
    if pdf_bytes is None:
        st.error("The User Guide PDF was not found in the application package.")
        return
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    st.subheader("📘 SiPREM User Guide")
    st.caption("The guide explains the input workbooks, linguistic variables, fuzzy numbers, SiPREM workflow, results, and visualizations.")
    html = '<iframe src="data:application/pdf;base64,' + b64 + '" width="100%" height="900" style="border:1px solid #d0d7de;border-radius:8px;"></iframe>'
    components.html(html, height=920, scrolling=True)
    st.download_button(
        "⬇️ Download User Guide (PDF)",
        data=pdf_bytes,
        file_name="SiPREM_User_Guide_EN.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


st.set_page_config(
    page_title="SiPREM — Military Equipment Requirement Prioritization",
    page_icon="🎯",
    layout="wide",
)


def save_uploaded(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".xlsx"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return Path(tmp.name)


def build_result_dataframe(data, result):
    return pd.DataFrame({
        "Rank": result["rank_siprem"].astype(int),
        "Requirement": data["ids"],
        "Description": data["descriptions"],
        "Phi_Plus": result["phi_plus"],
        "Phi_Minus": result["phi_minus"],
        "Phi_Net": result["phi_net"],
        "Category": categories_en(result["categories"]),
        "Impact_Divergence_Alert": (
            result["impact_matrix"].max(axis=1)
            - result["impact_matrix"].min(axis=1) >= 3
        ),
    }).sort_values(["Rank", "Requirement"]).reset_index(drop=True)


def build_excel(data, result):
    ranking = build_result_dataframe(data, result)
    phi_exp = pd.DataFrame(result["phi_net_experts"], index=data["ids"], columns=result["expert_names"])
    phi_exp.insert(0, "Requirement", phi_exp.index)
    impact = pd.DataFrame(result["impact_matrix"], index=data["ids"], columns=result["expert_names"])
    impact.insert(0, "Requirement", impact.index)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        ranking.to_excel(writer, sheet_name="Ranking", index=False)
        phi_exp.to_excel(writer, sheet_name="Phi_by_Specialist", index=False)
        impact.to_excel(writer, sheet_name="Mission_Impact", index=False)
        pd.DataFrame({
            "Criterion": criteria_en(),
            "Direction": ["Maximize", "Minimize", "Maximize", "Minimize", "Minimize"],
            "Preference_Type": ["Usual", "Level", "Level", "Level", "Level"],
        }).to_excel(writer, sheet_name="Configuration", index=False)
    output.seek(0)
    return output.getvalue()


def aggregate_weights(fahp):
    """Defuzzifica os pesos individuais e retorna média + distribuição por especialista."""
    wf = np.asarray(fahp["w_fuzzy_individual"], dtype=float)  # crit x 4 x exp
    crisp = np.array([[defuzz_coa(wf[k, :, e]) for e in range(wf.shape[2])] for k in range(wf.shape[0])])
    return crisp.mean(axis=1), crisp


def preference_relation_matrix(result, ids):
    """PROMETHEE I: P+ (domina), P- (dominado), I e R (incomparável)."""
    plus = np.asarray(result["phi_plus"], dtype=float)
    minus = np.asarray(result["phi_minus"], dtype=float)
    n = len(ids)
    rel = np.empty((n, n), dtype=object)
    eps = 1e-10
    for i in range(n):
        for j in range(n):
            if i == j:
                rel[i, j] = "I"
                continue
            pplus = plus[i] > plus[j] + eps
            pminus = minus[i] < minus[j] - eps
            dominated_plus = plus[i] < plus[j] - eps
            dominated_minus = minus[i] > minus[j] + eps
            if pplus and pminus:
                rel[i, j] = "P"
            elif dominated_plus and dominated_minus:
                rel[i, j] = "P-"
            elif abs(plus[i] - plus[j]) <= eps and abs(minus[i] - minus[j]) <= eps:
                rel[i, j] = "I"
            else:
                rel[i, j] = "R"
    return rel


def plot_weights(fahp):
    mean_w, crisp = aggregate_weights(fahp)
    names = criteria_en()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=mean_w, name="Mean weight"))
    for e, name in enumerate(fahp.get("sheets", [f"E{i+1}" for i in range(crisp.shape[1])])):
        fig.add_trace(go.Scatter(x=names, y=crisp[:, e], mode="markers", name=name))
    fig.update_layout(
        title="Fuzzy Criterion Weights Obtained by FAHP-Express",
        xaxis_title="Criterion", yaxis_title="Weight", legend_title="Source",
        height=500,
    )
    return fig


def _req_labels(ids):
    """IDs como categorias textuais para impedir que o Plotly trate R1/R2 ou 1/2 como eixo contínuo."""
    return [str(x) if str(x).startswith("R") else f"R{x}" for x in ids]


def plot_rank_heatmap(data, result):
    ranks = np.asarray(result["rank_experts"], dtype=float)
    labels = _req_labels(data["ids"])
    df = pd.DataFrame(ranks, index=labels, columns=result["expert_names"] )
    df["SIPREM"] = np.asarray(result["rank_siprem"], dtype=float)
    df = df.sort_values("SIPREM", ascending=True)

    fig = go.Figure(go.Heatmap(
        z=df.values,
        x=list(df.columns),
        y=list(df.index),
        text=np.vectorize(lambda x: f"{x:g}")(df.values),
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale="Blues_r",
        zmin=1, zmax=len(labels),
        xgap=1, ygap=1,
        colorbar=dict(title="Rank"),
        hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Ranking Position Heatmap: Specialists vs SiPREM",
        xaxis_title="Evaluation source",
        yaxis_title="Requirements",
        height=max(700, len(labels) * 24),
        margin=dict(l=70, r=40, t=70, b=70),
    )
    return fig


def plot_classification_heatmap(data, result):
    cats = categories_en(result["categories"])
    experts = result["expert_names"]
    labels = _req_labels(data["ids"])
    n_req = len(labels)
    class_mat = np.empty((n_req, len(experts)), dtype=object)
    for e in range(len(experts)):
        phi_e = result["phi_net_experts"][:, e]
        impact_e = result["impact_matrix"][:, e]
        key_e = np.where(impact_e == SCALE_MAX[0])[0]
        from core.classification import classify_requirements_kmeans
        ce, _, _, _ = classify_requirements_kmeans(phi_e, key_e, random_state=0)
        class_mat[:, e] = categories_en(ce)

    class_df = pd.DataFrame(class_mat, index=labels, columns=experts)
    class_df["SiPREM"] = cats
    order = np.argsort(result["rank_siprem"])
    class_df = class_df.iloc[order]

    mapping = {"KEY": 0, "Group 1": 1, "Group 2": 2, "Group 3": 3}
    z = class_df.replace(mapping).astype(float).values
    text = class_df.astype(str).values
    fig = go.Figure(go.Heatmap(
        z=z, x=list(class_df.columns), y=list(class_df.index), text=text,
        texttemplate="%{text}", textfont=dict(size=10),
        colorscale=[
            [0.00, "#3730a3"], [0.249, "#3730a3"],
            [0.250, "#2d9cdb"], [0.499, "#2d9cdb"],
            [0.500, "#7ac943"], [0.749, "#7ac943"],
            [0.750, "#fff200"], [1.00, "#fff200"],
        ],
        zmin=0, zmax=3, xgap=1, ygap=1,
        colorbar=dict(tickvals=[0,1,2,3], ticktext=["KEY","Group 1","Group 2","Group 3"], title="Classification"),
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Classification Heatmap by Source",
        xaxis_title="Source", yaxis_title="Requirements",
        height=max(700, n_req * 24), margin=dict(l=70, r=40, t=70, b=70),
    )
    return fig


def plot_boxplot(data, result):
    phi = np.asarray(result["phi_net_experts"], dtype=float)
    rows = []
    for j, rid in enumerate(data["ids"]):
        for e, name in enumerate(result["expert_names"]):
            rows.append({"Requirement": rid, "Net flow": phi[j, e], "Source": name})
    df = pd.DataFrame(rows)
    fig = px.box(df, x="Requirement", y="Net flow", points=False, title="Individual Net-Flow Boxplot by Requirement (SiPREM)")
    fig.add_trace(go.Scatter(
        x=data["ids"], y=result["phi_net"], mode="markers", marker=dict(symbol="star", size=11),
        name="Aggregated SiPREM net flow"
    ))
    fig.update_layout(height=550, xaxis_title="Requirements", yaxis_title="Φnet")
    return fig


def compute_unicriterion_net(data, result):
    """Calcula fluxos líquidos por requisito e critério, agregados entre especialistas."""
    n_req = data["n_requirements"]
    n_crit = data["n_criteria"]
    n_exp = data["n_experts"]
    q = result["q_vals"]
    p = result["p_vals"]
    uni = np.zeros((n_req, n_crit))

    for k in range(n_crit):
        exp_values = np.zeros((n_req, n_exp))
        for e, name in enumerate(result["expert_names"]):
            F = build_fuzzy_matrix(data["evaluations"][name])
            for i in range(n_req):
                sum_p = np.zeros(4)
                sum_m = np.zeros(4)
                for j in range(n_req):
                    if i == j:
                        continue
                    if CRIT_DIR[k] == 1:
                        D_ij = np.array([F[i, k, 0]-F[j, k, 3], F[i, k, 1]-F[j, k, 2], F[i, k, 2]-F[j, k, 1], F[i, k, 3]-F[j, k, 0]])
                    else:
                        D_ij = np.array([F[j, k, 0]-F[i, k, 3], F[j, k, 1]-F[i, k, 2], F[j, k, 2]-F[i, k, 1], F[j, k, 3]-F[i, k, 0]])
                    P = pref_trap(D_ij, int(PREF_TYPES[k]), q[k], p[k])
                    sum_p += P
                    # Pji is recomputed by reversed pair through the same criterion.
                    if CRIT_DIR[k] == 1:
                        D_ji = np.array([F[j, k, 0]-F[i, k, 3], F[j, k, 1]-F[i, k, 2], F[j, k, 2]-F[i, k, 1], F[j, k, 3]-F[i, k, 0]])
                    else:
                        D_ji = np.array([F[i, k, 0]-F[j, k, 3], F[i, k, 1]-F[j, k, 2], F[i, k, 2]-F[j, k, 1], F[i, k, 3]-F[j, k, 0]])
                    Pji = pref_trap(D_ji, int(PREF_TYPES[k]), q[k], p[k])
                    sum_m += Pji
                exp_values[i, e] = defuzz_coa(trap_scalar_div(sum_p, n_req - 1) - trap_scalar_div(sum_m, n_req - 1))
        uni[:, k] = exp_values.mean(axis=1)
    return uni


def plot_gaia(data, result):
    X = compute_unicriterion_net(data, result)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    total = np.sum(S ** 2)
    explained = (S[:2] ** 2) / total * 100 if total > 0 else np.array([0.0, 0.0])
    load = Vt[:2, :].T

    # Escala dos vetores. Os vetores representam a direção dos critérios no plano GAIA.
    max_coord = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), 1e-9)
    arrow_scale = max_coord * 0.22

    labels = _req_labels(data["ids"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode="markers",
        text=labels,
        customdata=np.column_stack([labels, result["phi_net"], result["rank_siprem"]]),
        marker=dict(
            size=10,
            color=result["phi_net"],
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(title="Φ"),
            line=dict(width=0.5, color="#444"),
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Φ líquido: %{customdata[1]:.4f}<br>"
            "Rank SiPREM: %{customdata[2]:.0f}<extra></extra>"
        ),
        name="Requirements",
    ))

    # Vetores dos critérios: cada um recebe uma cor própria, rótulo no extremo
    # e entrada na legenda. Isso evita a ambiguidade causada por várias setas pretas
    # convergindo para a mesma região da origem.
    criterion_colors = [
        "#1f77b4",  # azul
        "#d62728",  # vermelho
        "#2ca02c",  # verde
        "#9467bd",  # roxo
        "#ff7f0e",  # laranja
    ]
    short_names = CRITERIA_EN

    # Pequenos deslocamentos dos rótulos, proporcionais ao quadrante do vetor.
    # Assim os nomes não ficam todos exatamente sobre a ponta das setas.
    label_offsets = [
        (0.018, 0.018),
        (0.018, -0.022),
        (0.018, 0.024),
        (-0.020, -0.022),
        (-0.020, 0.022),
    ]

    for k, crit in enumerate(CRITERIA):
        vx = float(load[k, 0] * arrow_scale)
        vy = float(load[k, 1] * arrow_scale)
        color = criterion_colors[k % len(criterion_colors)]
        name = short_names.get(crit, crit)

        # Linha do vetor na legenda.
        fig.add_trace(go.Scatter(
            x=[0, vx], y=[0, vy],
            mode="lines",
            line=dict(color=color, width=3),
            name=name,
            hovertemplate=f"<b>{name}</b><br>Vetor do critério<extra></extra>",
            showlegend=True,
        ))

        # Cabeça da seta.
        fig.add_annotation(
            x=vx, y=vy, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.8,
            arrowcolor=color, text="",
        )

        # Nome do critério próximo à ponta, com fundo branco para leitura.
        ox, oy = label_offsets[k % len(label_offsets)]
        fig.add_annotation(
            x=vx + ox * max_coord,
            y=vy + oy * max_coord,
            xref="x", yref="y",
            text=f"<b>{name}</b>",
            showarrow=False,
            font=dict(size=11, color=color),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
        )

    # Identificação explícita da origem dos vetores.
    fig.add_annotation(
        x=0, y=0,
        xref="x", yref="y",
        text="<b>Origin</b>",
        showarrow=False,
        yshift=-18,
        font=dict(size=10, color="#555"),
        bgcolor="rgba(255,255,255,0.75)",
    )

    fig.add_hline(y=0, line_width=0.5, line_color="#cccccc")
    fig.add_vline(x=0, line_width=0.5, line_color="#cccccc")
    fig.update_layout(
        title=f"GAIA — Unicriterion Net-Flow Plane (Information Preserved: {explained[:2].sum():.2f}%)",
        xaxis_title=f"GAIA Axis 1 ({explained[0]:.1f}%)",
        yaxis_title=f"GAIA Axis 2 ({explained[1]:.1f}%)",
        height=700,
        margin=dict(l=70, r=60, t=80, b=70),
        hovermode="closest",
        legend=dict(
            title="Criterion vectors",
            orientation="v",
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
    )
    return fig

def plot_promethee_relation(data, result):
    rel = preference_relation_matrix(result, data["ids"])
    labels = _req_labels(data["ids"])
    mapping = {"P-": 0, "I": 1, "P": 2, "R": 3}
    z = np.vectorize(mapping.get)(rel)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        text=rel,
        texttemplate="%{text}",
        textfont=dict(size=7),
        colorscale=[
            [0.00, "#3333cc"], [0.249, "#3333cc"],
            [0.250, "#aaaaaa"], [0.499, "#aaaaaa"],
            [0.500, "#22aa22"], [0.749, "#22aa22"],
            [0.750, "#e8cc28"], [1.00, "#e8cc28"],
        ],
        zmin=0, zmax=3,
        xgap=1, ygap=1,
        colorbar=dict(
            tickvals=[0, 1, 2, 3],
            ticktext=["P− (Dominated)", "I (Indifferent)", "P (Dominates)", "R (Incomparable)"],
            title="Relation",
        ),
        hovertemplate="i=%{y}<br>j=%{x}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Fuzzy PROMETHEE I Relations in SiPREM",
        xaxis_title="Requirement j",
        yaxis_title="Requirement i",
        height=850,
        margin=dict(l=80, r=80, t=70, b=80),
    )
    return fig


# ---------------- UI ----------------
st.title("SiPREM")
st.subheader("System for Prioritizing Military Equipment Requirements")
st.caption("Web application — Python implementation of the current SiPREM mathematical core")

with st.sidebar:
    st.header("📥 1. Input files")
    show_template_downloads()
    st.divider()

    requirements_file = st.file_uploader(
        "Requirement assessment workbook", type=["xlsx"],
        help="Upload the requirement-assessment workbook used by the SiPREM model."
    )
    fahp_file = st.file_uploader(
        "FAHP-Express criteria weights workbook", type=["xlsx"],
        help="Upload the FAHP-Express workbook used to calculate criterion weights."
    )

    st.divider()
    st.header("▶ 2. Run")
    execute = st.button("▶ Run SiPREM", type="primary", use_container_width=True)

    st.divider()
    st.header("📘 User Guide")
    if st.button("Open User Guide", use_container_width=True):
        st.session_state["show_guide"] = True
    if st.button("Hide User Guide", use_container_width=True):
        st.session_state["show_guide"] = False

    st.divider()
    st.caption("Criteria configured in the current SiPREM model:")
    for c in criteria_en():
        st.write(f"• {c}")

if "show_guide" not in st.session_state:
    st.session_state["show_guide"] = False

if st.session_state["show_guide"]:
    show_user_guide()
    st.divider()

if "result" not in st.session_state:
    st.info("Upload both workbooks in the sidebar and click **Run SiPREM**.")

if execute:
    if requirements_file is None or fahp_file is None:
        st.error("Please select both workbooks before running SiPREM.")
        st.stop()
    req_path = save_uploaded(requirements_file)
    fahp_path = save_uploaded(fahp_file)
    try:
        with st.spinner("Running FAHP + Fuzzy PROMETHEE + classification..."):
            data = read_expert_workbook(req_path)
            fahp = read_fahp_workbook(fahp_path)
            result = run_siprem(data["evaluations"], fahp["w_fuzzy_individual"], random_state=0)
            ranking = build_result_dataframe(data, result)
            excel_bytes = build_excel(data, result)
        st.session_state.update(data=data, fahp=fahp, result=result, ranking=ranking, excel_bytes=excel_bytes)
        st.success("SiPREM completed successfully.")
    except Exception as exc:
        st.error("SiPREM could not be executed with the uploaded workbooks.")
        st.exception(exc)
        st.stop()

if "result" in st.session_state:
    data = st.session_state["data"]
    fahp = st.session_state["fahp"]
    result = st.session_state["result"]
    ranking = st.session_state["ranking"]

    st.divider()
    st.header("Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Specialists", data["n_experts"])
    c2.metric("Requirements", data["n_requirements"])
    c3.metric("Criteria", data["n_criteria"])
    c4.metric("KEY requirements", int(np.sum(ranking["Category"] == "KEY")))

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking", "📈 Charts", "👥 Specialists", "ℹ️ Inputs"])

    with tab1:
        st.dataframe(ranking, use_container_width=True, hide_index=True, column_config={
            "Phi_Plus": st.column_config.NumberColumn("Φ+", format="%.6f"),
            "Phi_Minus": st.column_config.NumberColumn("Φ−", format="%.6f"),
            "Phi_Net": st.column_config.NumberColumn("Φnet", format="%.6f"),
            "Impact_Divergence_Alert": st.column_config.CheckboxColumn("Mission Impact Divergence"),
        })
        st.download_button(
            "⬇️ Download Excel Results", data=st.session_state["excel_bytes"],
            file_name="SiPREM_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary"
        )

    with tab2:
        st.subheader("1. FAHP-Express Criterion Weights")
        st.plotly_chart(plot_weights(fahp), use_container_width=True)
        st.caption("Bars = mean defuzzified weights; points = individual specialist weights.")

        st.subheader("2. Fuzzy PROMETHEE I Relations")
        st.plotly_chart(plot_promethee_relation(data, result), use_container_width=True)
        st.caption("P = dominates; P− = dominated; I = indifferent; R = incomparable.")

        st.subheader("3. Ranking Position Heatmap")
        st.plotly_chart(plot_rank_heatmap(data, result), use_container_width=True)

        st.subheader("4. Classification Heatmap by Source")
        st.plotly_chart(plot_classification_heatmap(data, result), use_container_width=True)

        st.subheader("5. GAIA — Unicriterion Net Flows")
        with st.spinner("Calculating the GAIA plane..."):
            st.plotly_chart(plot_gaia(data, result), use_container_width=True)

        st.subheader("6. Individual Net-Flow Boxplot")
        st.plotly_chart(plot_boxplot(data, result), use_container_width=True)

    with tab3:
        phi_exp = pd.DataFrame(result["phi_net_experts"], index=data["ids"], columns=result["expert_names"])
        phi_exp.index.name = "Requirement"
        st.dataframe(phi_exp, use_container_width=True)
        st.markdown("**Mission Impact assessments reported by specialists**")
        impact = pd.DataFrame(result["impact_matrix"], index=data["ids"], columns=result["expert_names"])
        impact.index.name = "Requirement"
        st.dataframe(impact, use_container_width=True)

    with tab4:
        st.write("**Specialist worksheets found:**", ", ".join(data["sheets"]))
        st.write("**FAHP worksheets:**", ", ".join(fahp["sheets"]))
        st.table(pd.DataFrame({
            "Criterion": criteria_en(),
            "Direction": ["Maximize", "Minimize", "Maximize", "Minimize", "Minimize"],
            "Preference Type": ["Usual", "Level", "Level", "Level", "Level"],
        }))

st.divider()
st.caption("SiPREM — Web application. The mathematical calculation logic remains in the core/ module.")
