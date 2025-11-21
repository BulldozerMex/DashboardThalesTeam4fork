import streamlit as st
import base64

import streamlit as st

def theme_css(mode: str = "auto") -> str:
    """
    Returns a <style> block that:
      - Defines light and dark color tokens with CSS variables based on Thales Blue.
      - Supports 'auto' (prefers-color-scheme), 'light', and 'dark'.
      - Hides Streamlit Header and Footer.
    """
    
    # Colores base (Light Mode - Azul Pálido Sutil)
    light_mode_vars = """
    --bg:#a8d0ff;       /* Azul Pálido Sutil (Fondo principal) */
    --bg-alt: #6eb8f5;   /* Azul Claro pálido (Fondo secundario/Sidebar) */
    --fg: #000000ff;       /* Negro (Texto principal) */
    --muted: #000000ff;      /* Gris para texto 'muted' */
    --primary: #1ebcde;    /* Azul Thales Claro (Acento) */
    --border: #6c70a3ff;   /* Borde */
    """

    # Colores Dark Mode
    dark_mode_vars = """
    --bg: #08306b;       /* Azul Thales Oscuro (Fondo principal) */
    --bg-alt: #000000;     /* Negro (Fondo secundario/Sidebar) */
    --fg: #ffffff;         /* Blanco (Texto principal) */
    --muted: #a0a3ad;      /* Gris claro para texto 'muted' */
    --primary: #1ebcde;    /* Azul Thales Claro (Acento) */
    --border: #3b42a9ff;   /* Borde */
    """
    
    # Lógica para aplicar los colores según el 'mode'
    initial_vars = light_mode_vars
    auto_dark_css = ""
    
    if mode == "auto":
        auto_dark_css = f"""
        @media (prefers-color-scheme: dark) {{
          :root {{
            {dark_mode_vars}
          }}
        }}
        """
    elif mode == "dark":
        initial_vars = dark_mode_vars
    
    return f"""
    <style>
    :root {{
    {initial_vars}
    }}

    @media (prefers-reduced-motion: no-preference) {{
      [data-testid="stAppViewContainer"],
      [data-testid="stSidebar"] {{
        transition: background-color .2s ease, color .2s ease, border-color .2s ease;
      }}
    }}

    {auto_dark_css}

    /* =========================================
       OCULTAR ELEMENTOS DE STREAMLIT (HEADER/FOOTER)
       ========================================= */
    
    /* Ocultar Header (Barra superior con menú hamburguesa y 'Deploy') */
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
        box-shadow: none !important;
        border-bottom: none !important;
        /* Si quieres que desaparezca y el contenido suba, descomenta la linea de abajo: */
        /* display: none !important; */
    }}

    /* Ocultar Footer ('Made with Streamlit') */
    footer {{
        visibility: hidden;
        display: none !important;
    }}
    
    /* Ajustar el padding superior del contenido principal 
       para aprovechar el espacio si el header es transparente/oculto */
    .block-container {{
        padding-top: 2rem !important;
    }}

    /* =========================================
       ESTILOS DE TEMA PERSONALIZADO
       ========================================= */

    /* Main content area */
    [data-testid="stAppViewContainer"],
    .block-container {{
      background-color: var(--bg);
      color: var(--fg);
    }}

    a, .stMarkdown a {{
      color: var(--primary);
    }}

    /* Input fields (Text, Number, Selectbox, Multiselect) */
    div[data-baseweb="input"] input,
    textarea, .stTextInput input, .stNumberInput input, .stTextArea textarea,
    .stSelectbox div[role="combobox"], .stMultiSelect div[role="combobox"] {{
      background-color: var(--bg-alt) !important;
      color: var(--fg) !important;
      border: 1px solid var(--border) !important;
    }}

    /* Buttons */
    .stButton > button, .stDownloadButton > button {{
      background-color: var(--primary) !important; 
      color: var(--fg) !important; 
      border: 1px solid var(--primary) !important;
      border-radius: 5px;
      font-weight: bold;
    }}

    .stButton > button:hover,
    [data-testid="stSidebar"] .stButton > button:hover {{
      background-color: #1693b3 !important; 
      border-color: #1693b3 !important;
    }}

    hr {{
      border-color: var(--border);
    }}

    /* Sidebar styling */
    [data-testid="stSidebar"] {{
      background-color: var(--bg-alt);
      border-right: 1px solid var(--border);
      color: var(--fg);
    }}

    [data-testid="stSidebar"] * {{
      color: var(--fg) !important;
    }}

    [data-testid="stSidebar"] a {{
      color: var(--primary) !important;
    }}

    /* Sidebar Inputs */
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] .stSelectbox div[role="combobox"],
    [data-testid="stSidebar"] .stMultiSelect div[role="combobox"],
    [data-testid="stSidebar"] .stNumberInput input {{
      background-color: var(--bg) !important;
      color: var(--fg) !important;
      border: 1px solid var(--border) !important;
    }}

    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stDownloadButton > button {{
      background-color: var(--primary) !important;
      color: var(--fg) !important;
      border: 1px solid var(--primary) !important;
    }}

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
      color: var(--fg) !important;
    }}
    </style>
    """

# -----------------------------------------
# Inicializar estado de sesión
# -----------------------------------------
if "role" not in st.session_state:
    st.session_state.role = None

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "auto"

# -----------------------------------------
# Definir roles disponibles
# -----------------------------------------
ROLES = [None, "Socio Thales", "Agente Policiaco", "Visitante"]

# -----------------------------------------
# Función de inicio de sesión
# -----------------------------------------
def login():
    st.title("🚀 Welcome to the Team 4 Dashboard!")
    st.header("Inicio de sesión")
    
    # Selector de rol
    role = st.selectbox("Selecciona tu rol:", ROLES)

    if st.button("Entrar"):
        st.session_state.role = role
        st.rerun()

# -----------------------------------------
# Definir páginas (dentro del folder 'pages')
# -----------------------------------------
eda = st.Page(
    "pages/EDA.py",
    title="Análisis Exploratorio de Datos",
    icon=":material/analytics:",
)

predicciones = st.Page(
    "pages/Predicciones.py",
    title="Predicciones",
    icon=":material/trending_up:",
)

mapa = st.Page(
    "pages/Mapa.py",
    title="Mapa",
    icon=":material/map:",
)

chat = st.Page(
    "pages/Chat.py",
    title="Chat",
    icon=":material/chat:",
)

# -----------------------------------------
# Asignar accesos según el rol
# -----------------------------------------
page_dict = {
    "Comunes": [chat],
}

if st.session_state.role == "Socio Thales":
    page_dict.update({
        "EDA": [eda],
        "Predicciones": [predicciones],
        "Mapa": [mapa],
    })

elif st.session_state.role == "Agente Policiaco":
    page_dict.update({
        "Mapa": [mapa],
    })

elif st.session_state.role == "Visitante":
    page_dict.update({
        "Mapa": [mapa],
    })

# -----------------------------------------
# Selector de tema en el sidebar (siempre visible)
# -----------------------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("⚙️ Tema")
    
    theme_options = {
        "Auto": "auto",
        "Claro": "light",
        "Oscuro": "dark"
    }
    
    theme_choice = st.selectbox(
        "Modo de visualización:",
        options=list(theme_options.keys()),
        index=list(theme_options.values()).index(st.session_state.theme_mode),
        help="Cambia el tema de la aplicación"
    )
    
    # Actualizar el tema inmediatamente cuando cambia
    selected_theme = theme_options[theme_choice]
    if st.session_state.theme_mode != selected_theme:
        st.session_state.theme_mode = selected_theme
        st.rerun()

# =========================================
# INYECCIÓN DEL TEMA
# Se inyecta DESPUÉS de que el selector actualice el estado
# =========================================
st.markdown(theme_css(mode=st.session_state.theme_mode), unsafe_allow_html=True)

# -----------------------------------------
# Navegación principal
# -----------------------------------------
if len(page_dict) > 0 and st.session_state.role:
    pg = st.navigation(page_dict)
else:
    pg = st.navigation([st.Page(login)])

pg.run()