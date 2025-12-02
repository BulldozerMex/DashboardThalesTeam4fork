import duckdb
import streamlit as st
import pandas as pd
import numpy as np
import requests, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Chat Local (Ollama)", page_icon="📚")
st.title("📚 Chat  — 100% Local (Ollama)")

# ---------- Sidebar settings ----------
with st.sidebar:
    st.header("Configuración")
    model = st.text_input("Modelo Ollama", value="phi:latest",
                          help="Ejemplos: llama3, llama3:8b-instruct, phi3, mistral:instruct")
    top_k = st.slider("Número de filas k como contexto", 1, 10, 3)
    max_rows = st.number_input("Limite de filas (velocidad)", 100, 100000, 1000, step=100)
    temperature = st.slider("Temperatura", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Máximo de tokens nuevos", 32, 1024, 256, 32)
    if st.button("🔄 Resetear chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Haz empezado un nuevo chat"}]
        st.rerun()


@st.cache_data
def load_data():
    try:
        con = duckdb.connect("crimes_fgj.db", read_only=True)
        query = "SELECT * FROM crimes_raw LiMIT 1000"
        df = con.execute(query).df()
        con.close()
        return df
    except Exception as e:
        st.error(f"Error cargando la base de datos: {e}")
        return pd.DataFrame()


df = load_data()
st.success(f"Cargadas {len(df):,} filas × {len(df.columns)} columnas")
st.dataframe(df.head(10), use_container_width=True)

# Choose columns to index
text_cols = st.multiselect(
    "Columnas para crear texto que se pueda buscar (seleccione titulos/notas/descripción/campos clave):",
    options=list(df.columns),
    default=list(df.columns[: min(3, len(df.columns))])
)
if not text_cols:
    st.warning("Seleccione al menos una columna para el recuperador.")
    st.stop()

# ---------- Build TF-IDF retriever ----------
#a classic technique used in information retrieval and text-based search systems 
# to find and rank documents relevant to a query.
@st.cache_data(show_spinner=False)
def build_corpus_vectors(_df: pd.DataFrame, _cols):
    text_series = _df[_cols].astype(str).apply(lambda r: " | ".join(r.values), axis=1)
    vec = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(text_series.values)
    return text_series, vec, X

text_series, vectorizer, X = build_corpus_vectors(df, text_cols)

def retrieve(query: str, k: int):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

# ---------- Chat state ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Pregunta algo sobre tu archivo CSV y basaré mi respuesta en las filas coincidentes."}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Ollama call (local) ----------
OLLAMA_CHAT_URL = "http://localhost:11434/api/generate"  # using /generate for simple prompting

def stream_from_ollama(prompt: str):
    try:
        with requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            stream=True,
            timeout=0xFFFF,
        ) as r:
            r.raise_for_status()
            full = ""
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    token = data["response"]
                    full += token
                    yield token
                if data.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield "⚠️ Cannot reach Ollama at http://localhost:11434. Is `ollama serve` running?"
    except Exception as e:
        yield f"⚠️ Error: {e}"

# ---------- Prompt template ----------
SYSTEM_INSTRUCTION = (
    "Eres un asistente útil. Responde SO´LO usando las filas de contexto CSV proporcionadas. "
    "Si la respuesta no está en el contexto, indica que no la encuentras."
)

def build_prompt(user_q: str, rows_md: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"QUESTION:\n{user_q}\n\n"
        f"CONTEXT (CSV rows):\n{rows_md}\n\n"
        f"ANSWER:"
    )

# ---------- Handle user question ----------
if user_q := st.chat_input("Pregunta algo sobre el CSV…"):
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("BUscando columnas relevantes…"):
            idxs, scores = retrieve(user_q, top_k)
            top_rows = df.iloc[idxs]
            st.caption("Columnas Top (usadas como contexto):")
            st.dataframe(top_rows, use_container_width=True)

            # Compact context block
            rows_md = "\n".join(
                f"- ROW {i}: " + " | ".join(f"{c}={str(top_rows.iloc[i][c])}" for c in text_cols)
                for i in range(len(top_rows))
            )

        with st.spinner("Generando respuesta…"):
            prompt = build_prompt(user_q, rows_md)
            placeholder = st.empty()
            acc = ""
            for tok in stream_from_ollama(prompt):
                acc += tok
                placeholder.markdown(acc)

    st.session_state.messages.append({"role": "assistant", "content": acc})