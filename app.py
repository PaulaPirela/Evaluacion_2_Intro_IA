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
def get_chatbot_chain(_api_key):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", 
             """Eres 'Bio Gemini', un agente de IA experto en biología. Tu propósito es dar respuestas precisas y educativas.
             Reglas de Interacción:
             - **Tono**: Didáctico, científico y amigable.
             - **Precisión**: Prioriza la exactitud científica. Si no estás seguro, indícalo.
             - **Seguridad**: Nunca des consejos médicos o veterinarios; recomienda consultar a un profesional cualificado.
             - **Formato**: Usa Markdown (negritas, listas, etc.) para estructurar tus respuestas y mejorar la legibilidad."""),
            ("human", "{user_question}")
        ]
    )
    llm = ChatGroq(api_key=_api_key, model="llama3-70b-8192")
    return prompt_template | llm | StrOutputParser()

# --- GESTIÓN DE ESTADO DE SESIÓN ---
if "groq_api_key" not in st.session_state: st.session_state.groq_api_key = None
if "messages" not in st.session_state: st.session_state.messages = []
if "chain" not in st.session_state: st.session_state.chain = None

# --- PANTALLA DE BIENVENIDA / API KEY ---
if not st.session_state.groq_api_key:
    # El div 'api-container' ha sido eliminado de aquí.
    
    # Se crea un contenedor para centrar los elementos, pero sin el estilo de 'tarjeta'.
    col1, col2, col3 = st.columns([1, 2.5, 1])
    with col2:
        if logo_exists:
            st.image(logo_path, width=80)
        else:
            st.markdown("<p style='font-size: 60px; text-align: center;'>🧬</p>", unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>Bienvenido a Bio Gemini</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Para comenzar, necesitas una API Key de Groq.</p>", unsafe_allow_html=True)
        
        with st.form("api_key_form"):
            api_key_input = st.text_input(
                "Tu API Key de Groq", type="password", placeholder="gsk_xxxxxxxxxx", 
                help="Obtén tu clave gratuita en console.groq.com"
            )
            submitted = st.form_submit_button("Activar Bio Gemini")
            if submitted:
                if api_key_input and api_key_input.startswith("gsk_"):
                    st.session_state.groq_api_key = api_key_input
                    st.session_state.chain = get_chatbot_chain(st.session_state.groq_api_key)
                    st.rerun()
                else:
                    st.error("Por favor, ingresa una API Key de Groq válida que comience con 'gsk_'.")

# --- INTERFAZ DE CHAT PRINCIPAL ---
else:
    st.markdown("<h1 style='text-align: center; color: var(--accent-green);'>Bio Gemini</h1>", unsafe_allow_html=True)
    
    USER_AVATAR = "👤"
    BOT_AVATAR = logo_path if logo_exists else "✨"

    if not st.session_state.messages:
        st.markdown("<h2 style='text-align: center; color: var(--text-primary);'>¿Cómo puedo ayudarte hoy?</h2>", unsafe_allow_html=True)
        st.markdown("<div class='suggestion-buttons'>", unsafe_allow_html=True)
        cols = st.columns(2)
        suggestions = [
            "Explícame la fotosíntesis", "¿Qué es la edición genética CRISPR?",
            "Describe un animal abisal", "Diferencias: mitosis y meiosis"
        ]
        
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    st.session_state.selected_prompt = suggestion
                    st.rerun()
    
    for message in st.session_state.messages:
        avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    prompt = st.chat_input("Pregúntame algo de biología...", key="chat_input_main") or st.session_state.get("selected_prompt")

    if prompt:
        if "selected_prompt" in st.session_state: del st.session_state.selected_prompt
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            try:
                response = st.write_stream(st.session_state.chain.stream({"user_question": prompt}))
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = "Lo siento, ha ocurrido un error. Verifica tu API Key y la conexión."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
