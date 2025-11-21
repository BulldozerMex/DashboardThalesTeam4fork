import streamlit as st
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm 
import matplotlib.colors 
import matplotlib.ticker as ticker 
import matplotlib.patches as mpatches 
import seaborn as sns
import squarify
from scipy.stats import chi2_contingency
import numpy as np

# ===========================
# CONFIGURACIÓN DE LA PÁGINA Y ESTILOS CSS
# ===========================
st.set_page_config(page_title="EDA - Robos CDMX", layout="wide")

# INYECCIÓN DE CSS
st.markdown("""
<style>
    /* 1. SLIDER ROJO */
    div.stSlider > div[data-baseweb = "slider"] > div > div {
        background-color: #ff4b4b !important;
    }
    div.stSlider > div[data-baseweb = "slider"] > div > div > div {
        background-color: #ff4b4b !important;
    }

    /* 2. ACENTOS AZULES */
    .st-emotion-cache-16txtl3 {
        color: #6cd1ff !important;
    }

    /* 3. TARJETAS DE MÉTRICAS */
    div[data-testid="stMetric"] {
        background-color: white !important;
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 1px 1px 4px rgba(0,0,0,0.1);
    }
    
    /* 4. TÍTULOS DE MÉTRICAS -> NEGRO Y NEGRITAS */
    [data-testid="stMetricLabel"] * {
        color: #000000 !important;   
        font-weight: 900 !important; 
        font-size: 1.1rem !important;
    }
    
    /* VALORES DE MÉTRICAS -> GRIS OSCURO */
    [data-testid="stMetricValue"] * {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Análisis Estadístico Descriptivo")
st.markdown("""
Esta página muestra un resumen del análisis descriptivo y las pruebas **Chi-cuadrado** realizadas con los datos de robos por alcaldía y horario.

**Hipótesis:** 
* $H_0$: Las alcaldías centrales NO registran más robos en horario laboral comparado con las periféricas.
* $H_a$: Las alcaldías centrales registran más robos en horario laboral.
""")

# ===========================
# FUNCIÓN PARA ESTILAR TABLAS
# ===========================
def estilar_tabla(df):
    return df.style.set_properties(**{
        'background-color': '#6cd1ff', 
        'color': 'black',
        'border-color': 'white'
    })

# ===========================
# CARGA DE DATOS
# ===========================
@st.cache_data
def load_data():
    try:
        con = duckdb.connect("crimes_fgj.db", read_only=True)
        query = "SELECT * FROM crimes_raw WHERE delito ILIKE '%ROBO%'"
        df = con.execute(query).df()
        con.close()
        return df
    except Exception as e:
        st.error(f"Error cargando la base de datos: {e}")
        return pd.DataFrame()

df_robo = load_data()

if df_robo.empty:
    st.warning("No se cargaron datos. Verifica que el archivo 'crimes_fgj.db' esté en la carpeta.")
    st.stop()

# ===========================
# LIMPIEZA BÁSICA
# ===========================
df_robo["fecha_hecho"] = pd.to_datetime(df_robo["fecha_hecho"], errors="coerce")
df_robo["hora"] = pd.to_datetime(df_robo["hora_hecho"], errors="coerce").dt.hour
df_robo["alcaldia_hecho"] = df_robo["alcaldia_hecho"].astype(str).str.upper().str.strip()

valores_basura = ["NAN", "NONE", "NULL", "NAT", "", "DESCONOCIDO", "CDMX (INDETERMINADA)"]
df_robo = df_robo.dropna(subset=["alcaldia_hecho", "hora"])
df_robo = df_robo[~df_robo["alcaldia_hecho"].isin(valores_basura)]

df_robo["horario"] = pd.cut(
    df_robo["hora"],
    bins=[0, 6, 12, 18, 24],
    labels=["Madrugada", "Mañana", "Tarde", "Noche"],
    right=False
)

# ==============================================================================
# SECCIONES 1 y 2: VISUALIZACIONES LADO A LADO
# ==============================================================================
col_viz_1, col_viz_2 = st.columns(2)

# --- COLUMNA IZQUIERDA: ALCALDÍAS ---
with col_viz_1:
    st.subheader("1️⃣ Robos por Alcaldía")
    
    df_alcaldia = df_robo["alcaldia_hecho"].value_counts().reset_index()
    df_alcaldia.columns = ["alcaldia", "robos"]
    
    opcion_viz = st.selectbox("Tipo de Gráfico (Alcaldía):", ["Barras horizontales", "Heatmap", "Treemap"])
    color_azul = "#6cd1ff"

    if opcion_viz == "Barras horizontales":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=df_alcaldia, y="alcaldia", x="robos", ax=ax, color=color_azul)
        
        # EJE X con formato miles (20,000)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        # Etiquetas en barras con formato miles
        ax.bar_label(ax.containers[0], fmt='{:,.0f}', padding=3, fontsize=9)
        
        ax.set_title("Total de Robos por Alcaldía")
        st.pyplot(fig)

    elif opcion_viz == "Heatmap":
        df_heat = df_alcaldia.set_index("alcaldia")
        fig, ax = plt.subplots(figsize=(6, 8))
        # Heatmap formato miles con coma (fmt=",d")
        sns.heatmap(df_heat, annot=True, fmt=",d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title("Heatmap de robos por alcaldía")
        st.pyplot(fig)

    elif opcion_viz == "Treemap":
        fig, ax = plt.subplots(figsize=(10, 6))
        df_tree = df_alcaldia[df_alcaldia["robos"] > 0]
        
        # Paleta YlGnBu (Tonos azul/verde/aqua)
        cmap = matplotlib.cm.get_cmap('YlGnBu')
        mini, maxi = df_tree["robos"].min(), df_tree["robos"].max()
        norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
        colors = [cmap(norm(value)) for value in df_tree["robos"]]
        
        # Treemap solo con números formateados
        squarify.plot(sizes=df_tree["robos"], 
                      label=df_tree["robos"].apply(lambda x: f"{x:,}"), 
                      alpha=0.9, color=colors, pad=True, 
                      text_kwargs={'fontsize':9, 'color':'black', 'weight':'bold'})
        
        # Leyenda lateral externa
        legend_handles = [mpatches.Patch(color=cmap(norm(row['robos'])), label=row['alcaldia']) 
                          for index, row in df_tree.iterrows()]
        
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.axis("off")
        st.pyplot(fig)

# --- COLUMNA DERECHA: HORAS ---
with col_viz_2:
    st.subheader("2️⃣ Robos por Hora")
    
    alcaldias = ["Todas"] + sorted(df_robo["alcaldia_hecho"].unique())
    selected_alcaldia = st.selectbox("Filtrar por alcaldía (Hora):", alcaldias)
    
    df_filtrado = df_robo.copy() if selected_alcaldia == "Todas" else df_robo[df_robo["alcaldia_hecho"] == selected_alcaldia]
    df_filtrado = df_filtrado[df_filtrado["hora"].between(0, 23)]

    fig, ax = plt.subplots(figsize=(8, 6)) 
    sns.countplot(x="hora", data=df_filtrado, ax=ax, color=color_azul)
    
    # Ajuste de Ejes
    ax.tick_params(axis='x', labelsize=7)
    # Eje Y con formato miles
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    
    ax.set_title(f"Robos por hora del día ({selected_alcaldia})")
    ax.set_ylabel("Cantidad")
    st.pyplot(fig)


st.markdown("---")

# ==============================================================================
# SECCIÓN 3: TEST CHI-CUADRADO
# ==============================================================================
st.subheader("3️⃣ Test Chi-cuadrado por zona (central vs periférica)")

radio = st.slider("Radio para alcaldías centrales:", 8, 12, 10)

zonas = {
    10: {
        "central": ["CUAUHTEMOC", "VENUSTIANO CARRANZA", "IZTACALCO", "BENITO JUAREZ", "MIGUEL HIDALGO", "GAM", "AZCAPOTZALCO", "COYOACAN"],
        "periferica": ["ALVARO OBREGON", "IZTAPALAPA", "TLALPAN", "XOCHIMILCO", "MAGDALENA CONTRERAS", "CUAJIMALPA DE MORELOS", "TLAHUAC", "MILPA ALTA"]
    },
    8: {
        "central": ["CUAUHTEMOC", "VENUSTIANO CARRANZA", "IZTACALCO", "BENITO JUAREZ", "MIGUEL HIDALGO", "GAM"],
        "periferica": ["AZCAPOTZALCO", "COYOACAN", "ALVARO OBREGON", "IZTAPALAPA", "TLALPAN", "XOCHIMILCO", "MAGDALENA CONTRERAS", "CUAJIMALPA DE MORELOS", "TLAHUAC", "MILPA ALTA"]
    },
    12: {
        "central": ["CUAUHTEMOC", "VENUSTIANO CARRANZA", "IZTACALCO", "BENITO JUAREZ", "MIGUEL HIDALGO", "GAM", "AZCAPOTZALCO", "COYOACAN", "ALVARO OBREGON", "IZTAPALAPA"],
        "periferica": ["TLALPAN", "XOCHIMILCO", "MAGDALENA CONTRERAS", "CUAJIMALPA DE MORELOS", "TLAHUAC", "MILPA ALTA"]
    }
}

central = zonas.get(radio, zonas[10])["central"]
periferica = zonas.get(radio, zonas[10])["periferica"]

df_robo["zona"] = df_robo["alcaldia_hecho"].apply(lambda x: "Central" if x in central else ("Periferica" if x in periferica else "Otra"))
df_test = df_robo[df_robo["zona"].isin(["Central", "Periferica"])].copy()
df_test["periodo"] = df_test["hora"].apply(lambda h: "Laboral" if 8 <= h < 18 else "No Laboral")

contingency = pd.crosstab(df_test["zona"], df_test["periodo"])
chi2, p, dof, expected = chi2_contingency(contingency)

st.markdown("#### Resultados Estadísticos")
c1, c2, c3 = st.columns(3)
c1.metric("Chi²", f"{chi2:.2f}")
c2.metric("p-valor", f"{p:.5f}")
c3.metric("Grados de libertad", f"{dof}")
st.write("") 

col_chi_1, col_chi_2 = st.columns([1, 1])

with col_chi_1:
    st.write("**Tabla de Contingencia:**")
    st.dataframe(estilar_tabla(contingency))
    
    st.write("") 
    st.write("**Distribución de Robos Laborales (Donut):**")
    
    # DONUT CHART
    df_laboral = df_test[df_test["periodo"] == "Laboral"]
    total_laboral = len(df_laboral)
    conteo_zonas = df_laboral["zona"].value_counts()
    val_central = conteo_zonas.get("Central", 0)
    val_perif = conteo_zonas.get("Periferica", 0)
    sizes = [val_central, val_perif]
    labels = ['Central', 'Periférica']
    colors_donut = ['#08306b', '#1f6eb3'] # Azul oscuro / Azul claro

    fig_donut, ax_donut = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax_donut.pie(
        sizes, labels=labels, colors=colors_donut, autopct='%1.1f%%', 
        startangle=90, pctdistance=0.85, 
        wedgeprops=dict(width=0.4, edgecolor='white')
    )
    plt.setp(texts, size=10, weight="bold")
    plt.setp(autotexts, size=10, weight="bold", color="white")
    ax_donut.text(0, 0, f"Total\n{total_laboral:,}", ha='center', va='center', fontsize=11, fontweight='bold')
    ax_donut.set_title("Proporción Central vs Periférica", fontsize=12)
    st.pyplot(fig_donut)

with col_chi_2:
    st.write("**Mapa de Calor de la Muestra:**")
    fig_heat, ax_heat = plt.subplots(figsize=(6, 6)) 
    ax_heat.set_title("Heatmap de Contingencia (Zonas vs Horario)", fontsize=12, pad=15)
    # Heatmap con miles
    sns.heatmap(contingency, annot=True, fmt=",d", cmap="Blues", ax=ax_heat, cbar=False)
    st.pyplot(fig_heat)

st.markdown("---")

# ==============================================================================
# SECCIÓN 4: CONCLUSIÓN Y MEDIDOR (GAUGE)
# ==============================================================================
st.subheader("4️⃣ Conclusión de la Hipótesis")

col_conc_1, col_conc_2 = st.columns([1, 1])

alpha = 0.05
se_rechaza = p < alpha

tot_cen_abs = contingency.loc['Central'].sum() if 'Central' in contingency.index else 1
tot_per_abs = contingency.loc['Periferica'].sum() if 'Periferica' in contingency.index else 1
prop_central_laboral = contingency.loc['Central', 'Laboral'] / tot_cen_abs if tot_cen_abs > 0 else 0
prop_perif_laboral = contingency.loc['Periferica', 'Laboral'] / tot_per_abs if tot_per_abs > 0 else 0


# --- GRÁFICO DE MEDIDOR (GAUGE) TIPO MEDIA LUNA ---
def dibujar_medidor(rechaza_ho):
    # Canvas transparente y pequeño
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Semicirculo: 
    # Pie chart de Matplotlib dibuja en sentido antihorario desde 0 grados (derecha).
    # Queremos:
    # - Derecha (0 a 90 grados): Verde -> Slice 1
    # - Izquierda (90 a 180 grados): Rojo -> Slice 2
    # - Abajo (180 a 360): Invisible -> Slice 3
    
    colors = ['#2e7d32', '#c62828', 'none'] # Verde, Rojo, Transparente
    slices = [1, 1, 2] 
    
    wedges, _ = ax.pie(slices, colors=colors, startangle=0, counterclock=True, 
                       wedgeprops=dict(width=0.3, edgecolor='white'))
    
    # Quitamos borde a la parte invisible
    wedges[2].set_edgecolor('none')
    
    # TEXTO DENTRO DEL ARCO
    # Izquierda (Rojo) -> RECHAZA
    ax.text(-0.5, 0.5, "RECHAZA", ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    # Derecha (Verde) -> ACEPTA
    ax.text(0.5, 0.5, "ACEPTA", ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # FLECHA
    if rechaza_ho:
        # Rechaza -> Izquierda (Rojo)
        arrow_angle = 135
        texto_resultado = "Rechazo Ho"
    else:
        # Acepta -> Derecha (Verde)
        arrow_angle = 45
        texto_resultado = "Acepto Ho"
    
    angle_rad = np.deg2rad(arrow_angle)
    arrow_len = 0.7
    arrow_x = arrow_len * np.cos(angle_rad)
    arrow_y = arrow_len * np.sin(angle_rad)
    
    # Dibujar flecha
    ax.arrow(0, 0, arrow_x, arrow_y, head_width=0.05, head_length=0.1, fc='black', ec='black', width=0.015)
    
    # Centro
    ax.add_artist(plt.Circle((0, 0), 0.05, fc='black'))
    
    # Texto inferior
    ax.text(0, -0.2, texto_resultado, ha='center', va='top', fontsize=12, fontweight='bold')
    
    # Límites para cortar la mitad de abajo
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-1.1, 1.1)
    plt.axis('off')
    
    return fig

with col_conc_1:
    st.write("**Estado de la Hipótesis Nula:**")
    fig_gauge = dibujar_medidor(se_rechaza)
    # Renderizar con fondo transparente
    st.pyplot(fig_gauge, transparent=True)

with col_conc_2:
    st.write("**Interpretación del Resultado:**")
    
    if se_rechaza:
        st.markdown(f"####  **Se Rechaza la Ho** (p < 0.05)")
        
        if prop_central_laboral > prop_perif_laboral:
            st.markdown(f"""
            La prueba estadística indica diferencias significativas.
            
            Existe evidencia para afirmar que las alcaldías **CENTRALES** (Azul Oscuro)
            tienen una mayor tasa de robos en horario laboral comparado con las periféricas.
            """)
        else:
            st.markdown("""
            Hay diferencias significativas, pero la zona Central tiene **MENOS** robos laborales proporcionalmente.
            """)
    else:
        st.markdown(f"####  **No se rechaza la Ho**")
        st.markdown("""
        No hay evidencia estadística suficiente para diferenciar el comportamiento entre zonas.
        """)