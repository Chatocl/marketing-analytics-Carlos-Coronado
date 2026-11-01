"""Tarea de segmentación de clientes: ejecutar con streamlit run app.py."""
import io
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

st.set_page_config(page_title="Segmentación de clientes", page_icon="📊", layout="wide")
st.title("Segmentación de clientes · Marketing Analytics")
st.caption("Visor de grupos y comparación de cinco técnicas. Todos los cálculos se realizan localmente al ejecutar la app.")

NUMERICAS = ["ingreso_anual", "gasto_total", "compras_tienda", "compras_web",
            "compras_catalogo", "visitas_web_mes", "recencia_dias", "hijos_pequenos", "edad"]
CATEGORICAS = ["educacion", "estado_civil"]


def leer_csv(archivo):
    datos = pd.read_csv(archivo)
    # Los archivos originales de Orange pueden traer prefijos de tipo.
    datos.columns = [str(c).replace("mD#", "").replace("D#", "").replace("m#", "").strip() for c in datos.columns]
    # Soportar variantes de nombres originadas al exportar desde Orange.
    return datos


def perfil(datos, variable):
    df = datos.copy()
    df[variable] = df[variable].astype("string").fillna("Sin asignar")
    df["gasto_total"] = pd.to_numeric(df["gasto_total"], errors="coerce")
    g = df.groupby(variable, dropna=False, observed=True).agg(
        clientes=(variable, "size"), gasto_medio=("gasto_total", "mean"),
        gasto_total=("gasto_total", "sum")).reset_index()
    g["porcentaje_clientes"] = 100 * g["clientes"] / len(df)
    total_gasto = g["gasto_total"].sum()
    g["porcentaje_gasto"] = 100 * g["gasto_total"] / total_gasto if total_gasto else np.nan
    return g.rename(columns={variable: "Segmento"})


def obtener_matriz(df, columnas):
    faltan = [c for c in columnas if c not in df]
    if faltan:
        raise ValueError("Faltan variables numéricas: " + ", ".join(faltan))
    nums = df[columnas].apply(pd.to_numeric, errors="coerce")
    if nums.notna().sum().min() == 0:
        raise ValueError("Una o varias variables no contienen valores numéricos.")
    nums = nums.fillna(nums.median()).fillna(0)
    scaler = StandardScaler()
    return nums, scaler.fit_transform(nums)


def silueta(z, etiquetas):
    etiquetas = np.asarray(etiquetas)
    # DBSCAN marca el ruido con -1: no contamos ruido como un clúster.
    buenos = etiquetas != -1
    grupos = np.unique(etiquetas[buenos])
    if len(grupos) < 2 or buenos.sum() <= len(grupos):
        return np.nan
    try:
        return float(silhouette_score(z[buenos], etiquetas[buenos], sample_size=min(1500, buenos.sum()), random_state=42))
    except ValueError:
        return np.nan


def comparar(df, k, eps, min_pts, gamma):
    nums, z = obtener_matriz(df, NUMERICAS)
    resultados = {}
    resultados["k-means"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(z)
    resultados["Jerárquico (Ward)"] = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(z)
    resultados["DBSCAN"] = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(z)
    resultados["Mezcla gaussiana (GMM)"] = GaussianMixture(n_components=k, n_init=3, random_state=42).fit_predict(z)

    faltan = [c for c in CATEGORICAS if c not in df]
    if faltan:
        raise ValueError("K-prototypes necesita las columnas: " + ", ".join(faltan) + ".")
    try:
        from kmodes.kprototypes import KPrototypes
    except ImportError as exc:
        raise ValueError("Falta instalar kmodes: pip install kmodes") from exc
    categorias = df[CATEGORICAS].fillna("No informado").astype(str)
    matriz_mix = np.concatenate([z.astype(object), categorias.to_numpy(dtype=object)], axis=1)
    indices_categoricos = list(range(z.shape[1], matriz_mix.shape[1]))
    modelo = KPrototypes(n_clusters=k, init="Cao", n_init=3, gamma=gamma, random_state=42, verbose=0)
    resultados["k-prototypes"] = modelo.fit_predict(matriz_mix, categorical=indices_categoricos)

    tabla = []
    for nombre, etiquetas in resultados.items():
        labels = np.asarray(etiquetas)
        no_ruido = labels != -1
        unicos, cantidades = np.unique(labels[no_ruido], return_counts=True)
        tabla.append({"Técnica": nombre, "N.º grupos": len(unicos),
                      "Tamaños": ", ".join(str(n) for n in sorted(cantidades, reverse=True)),
                      "Ruido": int((~no_ruido).sum()), "Silueta": silueta(z, labels)})
    tabla_comparativa = pd.DataFrame(tabla)
    acuerdos = pd.DataFrame(index=resultados.keys(), columns=resultados.keys(), dtype=float)
    for a, la in resultados.items():
        for b, lb in resultados.items():
            # ARI evalúa similitud entre particiones, sin depender de etiquetas C1/C2/C3.
            # En DBSCAN el ruido se representa como una categoría adicional.
            acuerdos.loc[a, b] = adjusted_rand_score(la, lb)
    asignaciones = df.copy()
    for nombre, labels in resultados.items():
        asignaciones["seg_" + {"k-means":"kmeans", "Jerárquico (Ward)":"ward", "DBSCAN":"dbscan",
                               "Mezcla gaussiana (GMM)":"gmm", "k-prototypes":"kproto"}[nombre]] = ["Ruido" if int(v) == -1 else f"C{int(v)+1}" for v in labels]
    return tabla_comparativa, acuerdos, asignaciones

visor, comp = st.tabs(["Visor de grupos", "Comparador de técnicas"])

with visor:
    st.subheader("Visor de segmentos desde CSV")
    archivo = st.file_uploader("Sube clientes_con_cluster.csv o segmentos_notebook.csv", type="csv", key="visor")
    if archivo is not None:
        try:
            df = leer_csv(archivo)
            candidatos = [c for c in df if c.lower() in {"cluster", "seg_kmeans", "seg_gmm", "seg_kproto", "seg_ward", "seg_dbscan"}]
            candidatos += [c for c in df if c not in candidatos and ("cluster" in c.lower() or c.startswith("seg_"))]
            if not candidatos:
                st.error("El archivo debe contener una columna de grupos, como Cluster o seg_kmeans.")
            elif "gasto_total" not in df:
                st.error("El CSV necesita gasto_total en unidades originales para calcular el perfil.")
            else:
                variable = st.selectbox("Columna que define el segmento", candidatos)
                tabla = perfil(df, variable)
                a, b, c = st.columns(3)
                a.metric("Clientes", f"{len(df):,}")
                b.metric("Segmentos", len(tabla))
                c.metric("Gasto total (US$)", f"{pd.to_numeric(df['gasto_total'], errors='coerce').sum():,.2f}")
                fig = px.bar(tabla, x="Segmento", y="clientes", color="Segmento", text="clientes",
                             title="Cantidad de clientes por segmento")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(tabla.rename(columns={"clientes":"Clientes", "porcentaje_clientes":"% clientes",
                                                       "gasto_medio":"Gasto medio (US$)", "porcentaje_gasto":"% gasto total",
                                                       "gasto_total":"Gasto agregado (US$)"}).round(2), use_container_width=True, hide_index=True)
                if {"ingreso_anual", "gasto_total"}.issubset(df):
                    graf = df.copy()
                    graf[variable] = graf[variable].astype(str)
                    st.plotly_chart(px.scatter(graf, x="ingreso_anual", y="gasto_total", color=variable,
                                               hover_data=[c for c in ["id_cliente"] if c in graf],
                                               title="Ingreso anual frente a gasto total"), use_container_width=True)
                st.download_button("Descargar perfil CSV", tabla.to_csv(index=False).encode("utf-8-sig"),
                                   "perfil_segmentos.csv", "text/csv")
        except Exception as exc:
            st.error(f"No se pudo leer el CSV: {exc}")

with comp:
    st.subheader("Comparador de cinco técnicas")
    st.write("Sube el CSV original con las nueve variables numéricas, educación y estado civil. Se utiliza la misma base y la misma estandarización de variables numéricas para todas las técnicas.")
    archivo_comp = st.file_uploader("CSV de clientes para comparar", type="csv", key="comparador")
    co1, co2, co3, co4 = st.columns(4)
    with co1:
        k = st.slider("Grupos para k-means, Ward, GMM y k-prototypes", 2, 8, 3)
    with co2:
        eps = st.slider("DBSCAN eps", 0.2, 5.0, 1.5, 0.1)
    with co3:
        min_pts = st.slider("DBSCAN min_samples", 2, 20, 5)
    with co4:
        gamma = st.number_input("K-prototypes gamma", min_value=0.01, value=1.0, step=0.1)
    st.caption("En k-prototypes, gamma pondera las diferencias categóricas; el valor 1 es configurable, no una equivalencia automática con los demás métodos.")
    if archivo_comp is not None and st.button("Ajustar y comparar técnicas", type="primary"):
        try:
            datos = leer_csv(archivo_comp)
            with st.spinner("Ajustando los cinco modelos…"):
                comparacion, acuerdos, asignaciones = comparar(datos, k, eps, min_pts, gamma)
            st.markdown("#### Resultados por técnica")
            st.dataframe(comparacion.round(3), hide_index=True, use_container_width=True)
            st.caption("La silueta se calcula en las variables numéricas estandarizadas; para DBSCAN se excluye el ruido y se deja en blanco si hay menos de dos grupos. En k-prototypes es una referencia sobre su proyección numérica, no una validación completa de la distancia mixta.")
            st.markdown("#### Coincidencia entre técnicas (ARI)")
            st.plotly_chart(px.imshow(acuerdos.astype(float), zmin=-1, zmax=1,
                                      color_continuous_scale="RdBu", color_continuous_midpoint=0,
                                      text_auto=".2f", title="Índice Rand ajustado: 1 equivale a particiones idénticas"), use_container_width=True)
            st.caption("ARI compara asignaciones sin depender del número o nombre del clúster. En DBSCAN, ruido se trata como categoría adicional.")
            st.markdown("#### Perfiles y asignaciones")
            tecnica = st.selectbox("Técnica para visualizar", [c for c in asignaciones if c.startswith("seg_")])
            st.dataframe(perfil(asignaciones, tecnica).round(2), use_container_width=True, hide_index=True)
            st.download_button("Descargar asignaciones de las cinco técnicas", asignaciones.to_csv(index=False).encode("utf-8-sig"),
                               "comparacion_segmentaciones.csv", "text/csv")
        except Exception as exc:
            st.error(f"No se pudo completar la comparación: {exc}")

st.caption("Nota: los rótulos C1, C2 y C3 son arbitrarios y pueden cambiar entre ejecuciones; compara tamaños, perfiles y ARI, no solamente el nombre del clúster.")
