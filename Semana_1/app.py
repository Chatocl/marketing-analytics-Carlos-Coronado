import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Configuración inicial")

st.write("Primera prueba de uso de Streamlit y ambiente de MA2026")

gasto = st.slider(
    "Seleccione nivel de gasto en publicidad",
    10,
    200,
    50
)

variable_x = np.array([
    [10],
    [20],
    [30],
    [40],
    [50]
])

variable_y = np.array([
    15,
    25,
    35,
    45,
    55
])

modelo_lr = LinearRegression()
modelo_lr.fit(variable_x, variable_y)

if st.button("Predecir"):
    resultado = modelo_lr.predict([[gasto]])

    st.success(
        f"Las ventas proyectadas para una inversión de ${gasto} "
        f"son: ${resultado[0]:.2f}"
    )