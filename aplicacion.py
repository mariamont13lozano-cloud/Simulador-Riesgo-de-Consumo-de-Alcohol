import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

st.set_page_config(page_title="Simulador TFM María", layout="centered")

st.markdown("""
<style>
/* Sliders en rojo */
div[data-baseweb="slider"] [role="slider"] {
    background-color: #e53935 !important;
    border-color: #e53935 !important;
}
div[data-baseweb="slider"] > div > div:first-child {
    background: #e53935 !important;
}

/* Botón estilo imagen */
div.stButton > button {
    background-color: white;
    color: #333;
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 0.4rem 1.2rem;
    font-size: 1rem;
    cursor: pointer;
}
div.stButton > button:hover {
    background-color: #f5f5f5;
    border-color: #999;
}
</style>
""", unsafe_allow_html=True)

# ── Entrenar modelo con datos reales ─────────────────────────────────────────
@st.cache_resource
def entrenar_modelo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_excel(os.path.join(base_dir, "Database.xlsx"))
    df['ALCOHOL_RIESGO'] = (df['Walc'] > 3).astype(int)
    df = df.drop(['Dalc', 'Walc'], axis=1)
    df_proc = pd.get_dummies(df, drop_first=True)
    y = df_proc['ALCOHOL_RIESGO']
    X = df_proc.drop('ALCOHOL_RIESGO', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    m = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    m.fit(X_train, y_train)
    return m, list(X.columns), X_train.mean().to_dict()

modelo, feature_cols, medias = entrenar_modelo()

UMBRAL = 0.25

# ── Cabecera ──────────────────────────────────────────────────────────────────
st.title("Simulador de Riesgo de Consumo - TFM María")
st.write("Mueve los controles para ver cómo cambia la probabilidad de riesgo.")

# ── Sliders ───────────────────────────────────────────────────────────────────
goout     = st.slider("Salidas con amigos (1: Muy poco, 5: Mucho)", 1, 5, 2)
absences  = st.slider("Faltas de asistencia", 0, 93, 2)
health    = st.slider("Estado de salud (1: Muy mal, 5: Muy bien)", 1, 5, 4)
studytime = st.slider("Tiempo de estudio semanal (1: <2h, 4: >10h)", 1, 4, 1)

# ── Botón ─────────────────────────────────────────────────────────────────────
if st.button("Calcular Riesgo"):
    row = {col: medias.get(col, 0) for col in feature_cols}
    row['goout']     = goout
    row['absences']  = absences
    row['health']    = health
    row['studytime'] = studytime

    input_df = pd.DataFrame([row])[feature_cols]
    proba = modelo.predict_proba(input_df)[0][1]

    if proba >= UMBRAL:
        st.markdown(
            f'<div style="background:#fdecea; border:1px solid #f5c6cb; '
            f'border-radius:6px; padding:1rem; color:#7b1a1a; font-size:1rem;">'
            f'⚠️ Perfil de <strong>alto riesgo</strong>. Probabilidad: {proba:.2f}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="background:#edf7ed; border:1px solid #c3e6cb; '
            f'border-radius:6px; padding:1rem; color:#1e4620; font-size:1rem;">'
            f'✅ Perfil de <strong>bajo riesgo</strong>. Probabilidad: {proba:.2f}</div>',
            unsafe_allow_html=True
        )
