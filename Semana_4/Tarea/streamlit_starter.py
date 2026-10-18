"""
Laboratorio de Recomendación — PUNTO DE PARTIDA para su propia app de Streamlit.
Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus recomendadores:
elegir un cliente y un método, ver sus 3 recomendaciones con el "porque", y comparar los métodos
(hit rate y cobertura) — con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent



st.set_page_config(page_title="Recomendador — Laboratorio", page_icon="☕", layout="wide")

VERDE = "#00704A"
NARANJA = "#C4512F"
GRIS = "#9A988E"

def ruta_csv(nombre):
    rutas = [
        BASE_DIR / "data" / nombre,
        BASE_DIR.parent / "data" / nombre,
        BASE_DIR.parent.parent / "data" / nombre,
        BASE_DIR / nombre,
    ]

    for ruta in rutas:
        if ruta.is_file():
            return ruta

    raise FileNotFoundError(
        f"No se encontró {nombre}. "
        f"Se buscaron estas rutas: {[str(r) for r in rutas]}"
    )

@st.cache_data
def cargar():
    catalogo = pd.read_csv(ruta_csv("catalogo.csv"))
    hist = pd.read_csv(ruta_csv("ordenes_historial.csv"))
    hold = pd.read_csv(ruta_csv("ordenes_holdout.csv"))
    exp = pd.read_csv(ruta_csv("experimento.csv"))
    clientes = pd.read_csv(ruta_csv("clientes.csv"))
    productos = list(catalogo["item"])
    M = (pd.crosstab(hist["customer_id"], hist["item"]) > 0).astype(int).reindex(
        index=clientes["customer_id"], columns=productos, fill_value=0)
    X = M.values.astype(np.float32)
    H = (pd.crosstab(hold["customer_id"], hold["item"]) > 0).astype(int).reindex(
        index=M.index, columns=productos, fill_value=0).values
    NUEVO = H * (1 - X)                                   # probó algo NUEVO en el holdout
    NUEVO[:, productos.index("Agua")] = 0
    es_control = (exp.set_index("customer_id").loc[M.index, "grupo"] == "control").values
    return productos, M.index.values, X, NUEVO, es_control


productos, ids, X, NUEVO, es_control = cargar()
COL_AGUA = productos.index("Agua")
popularidad = X.sum(axis=0)

# TODO 1: la tabla de similitudes producto × producto (con la diagonal en 0), calculada UNA vez
S_it = cosine_similarity(X.T)
np.fill_diagonal(S_it, 0)


def puntajes(metodo, x_vec, K, indice=None):
    """Puntajes de los 29 productos para UN cliente (x_vec = su vector de 0/1)."""
    if metodo == "Popularidad":
        return popularidad.astype(float)
    if metodo == "Filtrado por ítem":
        # TODO 2: suma de las similitudes de cada producto con lo que el cliente ya compró
        return x_vec @ S_it
    # TODO 3: filtrado por cliente: similitud del cliente con TODOS los demás, sus K vecinos más parecidos
    #         y la suma de sus compras ponderada por esa similitud
    sims = cosine_similarity(x_vec[None, :], X)[0]

    if indice is not None:
        sims[indice] = 0

    vecinos = np.argsort(sims)[-K:]

    return sims[vecinos] @ X[vecinos]


def top3(scores, x_vec):
    s = np.where(x_vec > 0, -np.inf, scores).astype(float)   # no recomendar lo ya comprado
    s[COL_AGUA] = -np.inf                                     # ni el Agua
    return np.argsort(-s)[:3]


# --- Título ---
st.title("☕ Recomendador de productos")
st.caption(
    "Esta aplicación ayuda a identificar qué productos recomendar a cada cliente "
    "y comparar qué modelo ofrece mejores resultados, para tomar decisiones "
    "de marketing que mejoren las ventas y la experiencia del cliente."
)

# TODO 4: barra lateral con los controles interactivos (mínimo UNO): un selector del método
#         (st.selectbox) y un slider con el número de vecinos K (st.slider)
with st.sidebar:
    st.header("Parámetros")

    metodo = st.selectbox(
        "Seleccionar recomendador",
        ["Popularidad", "Filtrado por ítem", "Filtrado por cliente"]
    )

    K = st.slider(
        "Número de vecinos similares",
        min_value=5,
        max_value=100,
        value=30,
        step=5
    )

tab1, tab2 = st.tabs(["Recomendar", "Comparar métodos"])

with tab1:
    cid = st.number_input("ID de cliente", min_value=int(ids.min()), max_value=int(ids.max()), value=14, step=1)
    fila = int(np.where(ids == cid)[0][0])
    x_vec = X[fila].copy()
    comprados = [productos[j] for j in np.where(x_vec > 0)[0]]

    st.write("**Productos que ya compró:**")

    st.write(", ".join(comprados) if comprados else "Sin historial de compras")

    # TODO 5: muestren las 3 recomendaciones con su puntaje y, para cada una, el "porque"
    #         - por ítem: "porque compró X (similitud 0.87)"
    #         - por cliente: "N de sus K vecinos más parecidos lo compraron"
    #         - popularidad: "es de lo más comprado"
    #         (y qué hacen con un cliente SIN historial: los métodos colaborativos dan puntaje 0)
    scores = puntajes(metodo, x_vec, K, fila)

    # Respaldo para clientes sin historial
    if x_vec.sum() == 0 and metodo != "Popularidad":
        scores = popularidad.astype(float)
        st.info(
            "Cliente sin historial: se utiliza popularidad "
            "como respaldo."
        )

    recomendaciones = top3(scores, x_vec)

    columnas = st.columns(3)

    for i, j in enumerate(recomendaciones):

        producto = productos[j]
        puntaje = scores[j]

        if x_vec.sum() == 0 or metodo == "Popularidad":
            razon = "Es uno de los productos más comprados."

        elif metodo == "Filtrado por ítem":

            comprados = np.where((x_vec > 0) & (np.arange(len(x_vec)) != COL_AGUA))[0]

            mas_similar = comprados[
                np.argmax(S_it[comprados, j])
            ]

            razon = (
                f"Porque compró {productos[mas_similar]} "
                f"(similitud {S_it[mas_similar, j]:.2f})."
            )

        else:

            sims = cosine_similarity(
                x_vec[None, :], X
            )[0]

            sims[fila] = 0
            vecinos = np.argsort(sims)[-K:]

            cantidad = int(X[vecinos, j].sum())

            razon = (
                f"{cantidad} de sus {K} vecinos "
                "más parecidos compraron este producto."
            )

        with columnas[i]:
            with st.container(border=True):

                st.subheader(f"{i + 1}. {producto}")

                st.metric(
                    "Puntaje",
                    f"{puntaje:.3f}"
                )

                st.write("**¿Por qué?**")
                st.write(razon)

with tab2:
    # TODO 6: tabla de comparación (hit rate@3 y cobertura de cada método) calculada sobre los clientes del grupo
    #         CONTROL que probaron algo nuevo (NUEVO.sum(axis=1) > 0 y es_control), con al menos un control
    #         interactivo (por ejemplo, un slider con cuántos clientes evaluar o el K del filtrado por cliente).
    #         Pista: es la misma lógica de hit_rate_at_k del notebook.
    mask_eval = es_control & (NUEVO.sum(axis=1) > 0)

    indices = np.where(mask_eval)[0]

    n_evaluar = st.slider(
        "Cantidad de clientes a evaluar",
        min_value=50,
        max_value=len(indices),
        value=len(indices)
    )

    rng = np.random.default_rng(42)

    seleccion = rng.permutation(indices)[:n_evaluar]

    resultados = []

    for nombre in [
        "Popularidad",
        "Filtrado por ítem",
        "Filtrado por cliente"
    ]:

        aciertos = []
        productos_recomendados = []

        for u in seleccion:

            x_vec = X[u]

            scores = puntajes(nombre, x_vec, K, u)

            recomendados = top3(scores, x_vec)

            aciertos.append(
                NUEVO[u, recomendados].any()
            )

            productos_recomendados.extend(
                recomendados.tolist()
            )

        resultados.append({
            "Recomendador": nombre,
            "Hit rate@3": np.mean(aciertos),
            "Cobertura": len(set(productos_recomendados))
        })

    tabla = pd.DataFrame(resultados)

    st.subheader("Comparación de modelos")

    st.dataframe(
        tabla.style.format({
            "Hit rate@3": "{:.1%}",
            "Cobertura": "{:.0f}"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Hit rate@3")

    st.bar_chart(
        tabla.set_index("Recomendador")["Hit rate@3"]
    )

    st.subheader("Cobertura")

    st.bar_chart(
        tabla.set_index("Recomendador")["Cobertura"]
    )

