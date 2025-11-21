import streamlit as st
import pandas as pd
import numpy as np
import joblib
import duckdb
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="Predicción por Colonia", layout="wide")

st.title("🔍 Predicción de Robos: Colonia por Hora")
st.markdown("Distribución del riesgo predicho desglosado por **Alcaldía, Colonia y Hora**.")

# ==========================================
# 1. CARGA DE DATOS HISTÓRICOS (EL "DÓNDE")
# ==========================================
@st.cache_data
def load_historical_stats():
    """
    Carga estadísticas históricas para ponderar qué colonias son más peligrosas.
    """
    try:
        con = duckdb.connect("crimes_fgj.db", read_only=True)
        
        # Filtramos por 'TRANSEUNTE' para obtener los pesos de riesgo peatonal
        # Aseguramos que NO vengan nulos desde la base de datos
        query = """
            SELECT 
                alcaldia_hecho, 
                colonia_hecho, 
                COUNT(*) as total_robos
            FROM crimes_raw
            WHERE delito ILIKE '%TRANSEUNTE%' 
            AND alcaldia_hecho IS NOT NULL 
            AND colonia_hecho IS NOT NULL
            GROUP BY alcaldia_hecho, colonia_hecho
        """
        df = con.execute(query).df()
        con.close()
        
        # Limpieza adicional de Pandas para asegurar que no queden residuos
        df = df.dropna(subset=['alcaldia_hecho', 'colonia_hecho'])
        
        # Normalización de texto
        df['alcaldia_hecho'] = df['alcaldia_hecho'].astype(str).str.upper().str.strip()
        df['colonia_hecho'] = df['colonia_hecho'].astype(str).str.upper().str.strip()
        
        return df
    except Exception as e:
        st.error(f"Error conectando a la base de datos: {e}")
        return pd.DataFrame()

df_stats = load_historical_stats()

# ==========================================
# 2. CARGA DEL MODELO (EL "CUÁNDO")
# ==========================================
@st.cache_resource
def load_model():
    try:
        # Asegúrate de que este archivo exista en tu carpeta
        return joblib.load('xgboost_model.pkl')
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo 'xgboost_model.pkl'. Por favor súbelo.")
        return None
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

model = load_model()

# ==========================================
# 3. INTERFAZ DE USUARIO
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    fecha_sel = st.date_input("Fecha a predecir", datetime.now())

with col2:
    if not df_stats.empty:
        # --- CORRECCIÓN PARA ELIMINAR NAN ---
        unique_alcaldias = df_stats['alcaldia_hecho'].unique()
        
        # Filtro de lista por comprensión:
        # 1. pd.notna(x): Que no sea un objeto Nulo real
        # 2. str(x).strip() != "": Que no sea texto vacío
        # 3. "NAN" not in ...: Que no sea el texto literal "NAN"
        lista_alcaldias = [
            x for x in unique_alcaldias 
            if pd.notna(x) and str(x).strip() != "" and "NAN" not in str(x).upper()
        ]
        
        alcaldias_limpias = sorted(lista_alcaldias)
        
        alcaldia_sel = st.selectbox("Selecciona Alcaldía", alcaldias_limpias)
    else:
        alcaldia_sel = None
        st.warning("No hay datos de alcaldías disponibles.")

with col3:
    top_n = st.slider("Mostrar Top N Colonias más peligrosas", 5, 50, 10)


# ==========================================
# 4. LÓGICA DE PREDICCIÓN
# ==========================================
if st.button("Generar Matriz de Predicción"):
    if model is None:
        st.error("El modelo no está cargado.")
        st.stop()
        
    if alcaldia_sel is None:
        st.warning("Por favor selecciona una alcaldía.")
        st.stop()
    
    with st.spinner("Calculando riesgos..."):
        # -------------------------------------------------------
        # A) Obtener predicción temporal base (Curva de 24 horas)
        # -------------------------------------------------------
        input_data = []
        dia_sem = fecha_sel.weekday()
        
        # Features que tu modelo espera (IMPORTANTE: Deben coincidir con el entrenamiento)
        expected_features = ['año', 'mes', 'dia', 'hora', 'dia_semana', 
                             'sin_hora', 'cos_hora', 
                             'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'lag_24']

        for h in range(24):
            sin_h = np.sin(2 * np.pi * h / 24)
            cos_h = np.cos(2 * np.pi * h / 24)
            
            row = {
                "año": fecha_sel.year, 
                "mes": fecha_sel.month, 
                "dia": fecha_sel.day,
                "hora": h, 
                "dia_semana": dia_sem,
                "sin_hora": sin_h, 
                "cos_hora": cos_h,
                # Lags en 0 para predicción futura (Cold start)
                "lag_1": 0, "lag_2": 0, "lag_3": 0, "lag_6": 0, "lag_12": 0, "lag_24": 0
            }
            input_data.append(row)
        
        try:
            df_time_pred = pd.DataFrame(input_data)[expected_features]
            
            # Predicción base (Riesgo general horario)
            riesgo_base_horario = model.predict(df_time_pred)
            
            # ------------------------------------------------------
            # B) Obtener pesos espaciales (Distribución por Colonia)
            # ------------------------------------------------------
            # Filtramos colonias de la alcaldía seleccionada
            df_local = df_stats[df_stats['alcaldia_hecho'] == alcaldia_sel].copy()
            
            if df_local.empty:
                st.warning(f"No hay datos históricos de robos a transeúnte para {alcaldia_sel}.")
                st.stop()
            
            # Calculamos el peso de cada colonia
            total_crimes_local = df_local['total_robos'].sum()
            df_local['peso'] = df_local['total_robos'] / total_crimes_local
            
            # Top N colonias
            df_top_colonias = df_local.sort_values('total_robos', ascending=False).head(top_n)
            
            # ------------------------------------------------------
            # C) Cruzar Datos: Matriz Colonia x Hora
            # ------------------------------------------------------
            matrix_data = {}
            
            for _, row_colonia in df_top_colonias.iterrows():
                colonia_name = row_colonia['colonia_hecho']
                peso_colonia = row_colonia['peso']
                
                # Fórmula: Riesgo Base (Modelo) * Peso Histórico (Datos) * Escalar
                prediccion_colonia = riesgo_base_horario * peso_colonia * 100 
                matrix_data[colonia_name] = prediccion_colonia

            # Crear DataFrame final para Heatmap
            df_heatmap = pd.DataFrame(matrix_data).T 
            df_heatmap.columns = [f"{h}:00" for h in range(24)]
            
            # ==========================================
            # 5. VISUALIZACIÓN
            # ==========================================
            st.subheader(f"🔥 Mapa de Calor de Riesgo: {alcaldia_sel}")
            st.caption(f"Mostrando las {top_n} colonias con mayor incidencia histórica.")
            
            # Ajuste dinámico de altura
            fig_height = max(6, top_n * 0.5)
            fig, ax = plt.subplots(figsize=(14, fig_height))
            
            # Heatmap
            sns.heatmap(df_heatmap, cmap="inferno", annot=False, linewidths=.5, ax=ax)
            
            plt.xlabel("Hora del Día")
            plt.ylabel("Colonia")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Tabla de datos con FORMATO A UN DECIMAL
            with st.expander("📂 Ver datos detallados en tabla"):
                # AQUI ESTÁ EL CAMBIO: .format("{:.1f}")
                st.dataframe(
                    df_heatmap.style.background_gradient(cmap="Reds", axis=None).format("{:.1f}")
                )

        except Exception as e:
            st.error(f"Error durante la generación de predicciones: {e}")