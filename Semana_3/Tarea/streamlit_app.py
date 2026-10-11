
"""
Laboratorio Final — Starbucks Rewards
PUNTO DE PARTIDA para su propia app de Streamlit. Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus
resultados del laboratorio — el Qini de su T-learner y la tabla de políticas —
con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Laboratorio Final — Starbucks Rewards", page_icon="☕", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]


@st.cache_data
def load_data():
    ruta = Path(__file__).resolve().parent / "data" / "uplift_campaign.csv"
    return pd.read_csv(ruta)


@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, test_size: float, seed: int):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["treatment_group"])

    # TODO: entrenen el T-learner (dos modelos, igual que en el notebook)
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    modelo_tratado = LogisticRegression(max_iter=1000).fit(train_t[FEATURES], train_t["responded_60d"])
    modelo_control = LogisticRegression(max_iter=1000).fit(train_c[FEATURES], train_c["responded_60d"])

    test = test.copy()
    X_test = test[FEATURES]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    return train, test


def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)


# --- Título y carga de datos ---
st.title("Laboratorio Final — Starbucks Rewards")
st.caption("Identifica qué porcentaje de clientes conviene contactar para maximizar las ganancias de la campaña.")

df = load_data()

# TODO: agreguen un slider en la barra lateral para el % de la base a contactar
with st.sidebar:
    st.header("Parámetros")
    pct_contactar = st.slider("Porcentaje de clientes a contactar", 10, 100, 20, step=10)
    

train, test = entrenar_t_learner(df, test_size=0.30, seed=42)

# --- Curva Qini del T-learner ---

def curva_qini(test_df, score_col):
    orden = test_df.sort_values(
        score_col, ascending=False
    ).reset_index(drop=True)

    xs = [0]
    ys = [0]

    for pct in range(10, 101, 10):
        n = int(len(orden) * pct / 100)
        seleccion = orden.head(n)

        tratados = seleccion[
            seleccion["treatment_group"] == "treatment"
        ]

        control = seleccion[
            seleccion["treatment_group"] == "control"
        ]

        tasa_t = tratados["responded_60d"].mean()
        tasa_c = control["responded_60d"].mean()

        qini = (tasa_t - tasa_c) * n

        xs.append(pct)
        ys.append(qini)

    return xs, ys


xs, ys = curva_qini(test, "uplift_estimado")

st.subheader("Curva Qini del T-learner")

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(xs, ys, color=GREEN, label="T-learner")
ax.plot([0, 100], [0, ys[-1]], "--",
        color=GRAY, label="Al azar")

# --- Punto del porcentaje seleccionado ---
indice = xs.index(pct_contactar)
qini_seleccionado = ys[indice]

ax.scatter(
    pct_contactar,
    qini_seleccionado,
    color=ORANGE,
    s=100,
    zorder=5,
    label="Porcentaje seleccionado"
)

ax.axvline(
    x=pct_contactar,
    color=ORANGE,
    linestyle=":",
    alpha=0.6
)

ax.set_xlabel("Porcentaje de clientes contactados")
ax.set_ylabel("Respuestas incrementales")
ax.legend()
ax.grid(True)

st.pyplot(fig)


# --- Tabla de políticas ---

resultados = []

for pct in range(10, 101, 10):

    n = int(len(test) * pct / 100)

    seleccionados = test.sort_values(
        "uplift_estimado", ascending=False
    ).head(n).index

    valor_cliente, cantidad = valor_de_la_politica(
        test, seleccionados
    )

    resultados.append({
        "Porcentaje": pct,
        "Clientes": cantidad,
        "Valor por cliente": valor_cliente,
        "Valor total": valor_cliente * cantidad
    })

tabla_politicas = pd.DataFrame(resultados)

st.subheader("Tabla de políticas")

# --- Resaltar fila seleccionada ---
def resaltar_fila(fila):
    if fila["Porcentaje"] == pct_contactar:
        return ["background-color: #DFF2E6"] * len(fila)
    return [""] * len(fila)

st.dataframe(
    tabla_politicas.round(2).style.apply(resaltar_fila, axis=1),
    hide_index=True
)


# --- Resultado del porcentaje seleccionado ---

seleccion = tabla_politicas[
    tabla_politicas["Porcentaje"] == pct_contactar
].iloc[0]

st.subheader("Resultado de la política seleccionada")

st.write("Porcentaje a contactar:", pct_contactar, "%")
st.write("Clientes:", int(seleccion["Clientes"]))
st.write("Valor por cliente: $",
         round(seleccion["Valor por cliente"], 2))
st.write("Valor total: $",
         round(seleccion["Valor total"], 2))
