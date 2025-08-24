import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# --- CONFIGURACIÓN DE LA PÁGINA Y DEL LOGO ---
logo_path = "logo.png"

# Verifica si el archivo del logo existe
logo_exists = os.path.exists(logo_path)

# Configura la página con el logo o un emoji de respaldo
st.set_page_config(
    page_title="Bio Gemini",
    page_icon=logo_path if logo_exists else "🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- INYECCIÓN DE CSS PARA UN DISEÑO TEMÁTICO DE BIOLOGÍA ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

/* --- COLORES TEMÁTICOS --- */
:root {
    --primary-bg: #131314;      /* Negro espacial */
    --secondary-bg: #1e1f20;    /* Gris oscuro */
    --accent-green: #79F8B7;     /* Verde menta brillante */
    --accent-green-dark: #00A67E; /* Verde oscuro para botones */
    --text-primary: #e3e3e3;     /* Texto principal claro */
    --border-color: #3c4043;
}

/* --- GENERALES --- */
body {
    font-family: 'Google Sans', sans-serif, system-ui;
    background-color: var(--primary-bg) !important;
    color: var(--text-primary);
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stBottomBlockContainer"] {
    background-color: var(--primary-bg);
}
[data-testid="stSidebar"], [data-testid="stHeader"], #MainMenu, footer {
    display: none;
}

/* --- PANTALLA DE API KEY (SIN CONTENEDOR) --- */
/* Se mantiene el estilo del botón y el input, pero ya no hay una tarjeta contenedora */
.stButton button {
    background-color: var(--accent-green-dark);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    width: 100%;
    transition: background-color 0.3s ease;
}
.stButton button:hover {
    background-color: var(--accent-green);
    color: var(--primary-bg);
}
.stTextInput label {
    color: var(--text-primary) !important;
}

/* --- INTERFAZ DE CHAT --- */
[data-testid="stAppViewContainer"] {
    padding-top: 1rem;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"] {
    background: none; padding: 0; margin: 0; animation: fadeIn 0.5s ease-out;
}
.stChatMessage {
    max-width: 768px; margin: 0 auto 2rem auto;
}
[data-testid="stChatMessage"] [data-testid="stAvatar"] img {
    width: 40px; height: 40px;
}
.st-emotion-cache-1c7y2kd {
    font-weight: 600; color: var(--text-primary); padding-bottom: 8px;
}

[data-testid="stChatMessage"] > div[data-testid="stMarkdown"] {
    padding: 1.2em; border-radius: 12px; line-height: 1.6;
}
[data-testid="stChatMessage"][data-testid="chat-message-assistant"] > div[data-testid="stMarkdown"] {
    background-color: var(--secondary-bg);
    border: 1px solid var(--border-color);
    color: var(--accent-green);
}
[data-testid="stChatMessage"][data-testid="chat-message-user"] > div[data-testid="stMarkdown"] {
    background-color: #2a2b32;
    border: 1px solid #4a4d52;
    color: var(--accent-green);
}
pre {
    background-color: #0d0d0d; border-radius: 8px; padding: 1em;
    font-size: 0.9em; overflow-x: auto;
}
code {
    color: #c9d1d9; font-family: 'Fira Code', 'Courier New', monospace;
}
[data-testid="stChatInput"] {
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    background-color: rgba(30, 31, 32, 0.9); backdrop-filter: blur(10px);
    border-radius: 24px; padding: 8px 12px; border: 1px solid var(--border-color);
    width: 90%; max-width: 768px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
[data-testid="stChatInput"] textarea {
    background: none; color: var(--text-primary);
}
.suggestion-buttons {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px; max-width: 768px; margin: 1rem auto 2rem auto;
}
.suggestion-buttons .stButton button {
    width: 100%; text-align: left; background-color: var(--secondary-bg);
    border: 1px solid var(--border-color); color: var(--text-primary); padding: 12px;
    border-radius: 8px; transition: background-color 0.2s ease, border-color 0.2s ease;
}
.suggestion-buttons .stButton button:hover {
    background-color: #2a2b32;
    border-color: var(--accent-green);
}
</style>
""", unsafe_allow_html=True)


# --- LÓGICA DEL AGENTE (sin cambios) ---
@st.cache_resource
def get_chatbot_chain(_api_
