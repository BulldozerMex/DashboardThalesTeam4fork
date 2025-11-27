import streamlit as st
import pandas as pd
import numpy as np
import joblib
import duckdb
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import zlib

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="Predicción por Colonia", layout="wide")

st.title("🔍 Predicción de Crimenes: Colonia por Hora")
st.markdown("Distribución del riesgo predicho desglosado por **Alcaldía, Colonia y Hora**.")

# ==========================================
# 0. CONFIGURACIÓN DE LOS 3 TIPOS DE ROBOS
# ==========================================
DELITO_CONFIG = {
    "Robo a Transeúnte": {
        "sql_filter": "%TRANSEUNTE%",
        "model_file": "xgboost_model.pkl",
        "type": "temporal_base"
    },
    "Robo a Negocio": {
        "sql_filter": "%NEGOCIO%",
        "model_file": "model_neg_tran.pkl",
        "type": "spatiotemporal"
    },
    "Robo a Transporte": {
        "sql_filter": "%TRANSPORTE%",
        "model_file": "model_neg_tran.pkl",
        "type": "spatiotemporal"
    },
    "Homicidio y Feminicidio": {
        "sql_filter": "%HOMICIDIO%",
        "model_file": "model_hom_fem.pkl",
        "type": "spatiotemporal"
    },
    "Violación": {
        "sql_filter": "%VIOLACION%",
        "model_file": "model_violacion.pkl",
        "type": "spatiotemporal"
    }
}

# ==========================================
# 1. MENÚ PRINCIPAL
# ==========================================
tipo_delito = st.selectbox(
    "📂 Selecciona el Tipo de Delito:",
    list(DELITO_CONFIG.keys()), 
    index=0
)

current_config = DELITO_CONFIG[tipo_delito]

# ==========================================
# 2. CARGA DE DATOS Y MODELO
# ==========================================
@st.cache_data
def load_historical_stats(keyword_filter):
    try:
        con = duckdb.connect("crimes_fgj.db", read_only=True)
        query = f"""
            SELECT 
                alcaldia_hecho, 
                colonia_hecho, 
                COUNT(*) as total_robos
            FROM crimes_raw
            WHERE delito ILIKE '{keyword_filter}' 
            AND alcaldia_hecho IS NOT NULL 
            AND colonia_hecho IS NOT NULL
            GROUP BY alcaldia_hecho, colonia_hecho
        """
        df = con.execute(query).df()
        con.close()
        
        df = df.dropna(subset=['alcaldia_hecho', 'colonia_hecho'])
        df['alcaldia_hecho'] = df['alcaldia_hecho'].astype(str).str.upper().str.strip()
        df['colonia_hecho'] = df['colonia_hecho'].astype(str).str.upper().str.strip()
        return df
    except Exception as e:
        st.error(f"Error conectando a la base de datos: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model(filename):
    try:
        loaded = joblib.load(filename)
        if isinstance(loaded, str):
            st.error(f"🚨 El archivo {filename} es texto. Revisa el pkl.")
            return None
        return loaded
    except FileNotFoundError:
        st.warning(f"⚠️ Falta archivo: {filename}")
        return None
    except Exception as e:
        st.error(f"Error cargando {filename}: {e}")
        return None

df_stats = load_historical_stats(current_config["sql_filter"])
model = load_model(current_config["model_file"])

# ==========================================
# 3. FILTROS
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    fecha_sel = st.date_input("Fecha a predecir", datetime.now())

with col2:
    if not df_stats.empty:
        unique_alcaldias = sorted([x for x in df_stats['alcaldia_hecho'].unique() if x and "NAN" not in str(x)])
        alcaldia_sel = st.selectbox("Selecciona Alcaldía", unique_alcaldias)
    else:
        alcaldia_sel = None
        st.warning("No hay datos históricos.")

with col3:
    top_n = st.slider("Número de Colonias", 5, 50, 10)

# ==========================================
# 4. LÓGICA DE PREDICCIÓN
# ==========================================
def get_colonia_code(nombre_colonia):
    # Hash provisional
    return zlib.crc32(nombre_colonia.encode('utf-8')) % 1000

if st.button(f"Generar Mapa para {tipo_delito}"):
    
    if model is None or alcaldia_sel is None:
        st.stop()
    
    with st.spinner("Calculando..."):
        try:
            df_local = df_stats[df_stats['alcaldia_hecho'] == alcaldia_sel].copy()
            if df_local.empty:
                st.warning(f"No hay datos para {alcaldia_sel}.")
                st.stop()
            
            df_top_colonias = df_local.sort_values('total_robos', ascending=False).head(top_n)
            
            matrix_data = {}
            dia_sem = fecha_sel.weekday()
            mes = fecha_sel.month
            anio = fecha_sel.year

            # --- MODELO NUEVO (Negocio / Transporte) ---
            if current_config["type"] == "spatiotemporal":
                for _, row in df_top_colonias.iterrows():
                    col_name = row['colonia_hecho']
                    col_code = get_colonia_code(col_name)
                    
                    input_rows = []
                    for h in range(24):
                        input_rows.append({
                            "hora": h,
                            "dia_semana": dia_sem,
                            "mes": mes,
                            "colonia_code": col_code
                        })
                    
                    df_pred = pd.DataFrame(input_rows)
                    preds = model.predict(df_pred)
                    
                    # 1. CAMBIO IMPORTANTE: Quitamos la multiplicación * 100
                    # Usamos el valor crudo del modelo
                    matrix_data[col_name] = preds 

            # --- MODELO ANTIGUO (Transeúnte) ---
            else:
                input_data = []
                features_old = ['año', 'mes', 'dia', 'hora', 'dia_semana', 
                                'sin_hora', 'cos_hora', 
                                'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'lag_24']
                
                for h in range(24):
                    sin_h = np.sin(2 * np.pi * h / 24)
                    cos_h = np.cos(2 * np.pi * h / 24)
                    input_data.append({
                        "año": anio, "mes": mes, "dia": fecha_sel.day,
                        "hora": h, "dia_semana": dia_sem,
                        "sin_hora": sin_h, "cos_hora": cos_h,
                        "lag_1": 0, "lag_2": 0, "lag_3": 0, "lag_6": 0, "lag_12": 0, "lag_24": 0
                    })
                
                df_time = pd.DataFrame(input_data)[features_old]
                riesgo_base = model.predict(df_time)
                
                total_crimes = df_local['total_robos'].sum()
                for _, row in df_top_colonias.iterrows():
                    peso = row['total_robos'] / total_crimes
                    # Mantenemos *100 aquí porque el riesgo base suele ser muy pequeño (0.00x)
                    matrix_data[row['colonia_hecho']] = riesgo_base * peso * 100

            # --- VISUALIZACIÓN ---
            df_heatmap = pd.DataFrame(matrix_data).T 
            df_heatmap.columns = [f"{h}:00" for h in range(24)]
            
            st.subheader(f"🔥 Mapa de Calor: {tipo_delito}")
            
            fig_height = max(6, top_n * 0.5)
            fig, ax = plt.subplots(figsize=(14, fig_height))
            
            # 2. CAMBIO IMPORTANTE: Quitamos vmin=0
            # Dejamos que seaborn calcule el min y max automáticamente (Auto-contraste)
            sns.heatmap(
                df_heatmap, 
                cmap="magma",     
                annot=False, 
                linewidths=.5, 
                ax=ax
            )
            
            plt.xlabel("Hora del Día")
            plt.ylabel("Colonia")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            with st.expander("📂 Ver datos numéricos"):
                # Usamos un formato flexible
                st.dataframe(df_heatmap.style.background_gradient(cmap="magma", axis=None))

        except Exception as e:
            st.error(f"Error en visualización: {e}")