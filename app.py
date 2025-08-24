import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# --- CONFIGURACIÓN DE LA PÁGINA Y DEL LOGO ---
logo_path = "logo.png"
logo_exists = os.path.exists(logo_path)

st.set_page_config(
    page_title="Bio Gemini",
    page_icon=logo_path if logo_exists else "🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- INYECCIÓN DE CSS PARA EL DISEÑO FINAL ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

/* --- COLORES TEMÁTICOS --- */
:root {
    --primary-bg: #1e1e1e;      /* Gris oscuro para el fondo */
    --secondary-bg: #2b2b2b;    /* Gris un poco más claro para elementos */
    --user-bg: #EAEAEA;         /* Gris claro para el usuario */
    --accent-green: #79F8B7;     /* Verde menta brillante */
    --accent-green-dark: #00A67E; /* Verde oscuro para botones */
    --text-light: #FFFFFF;       /* Texto blanco puro */
    --text-dark: #000000;        /* Texto negro */
    --border-color: #3c4043;
}

/* --- GENERALES --- */
body {
    font-family: 'Google Sans', sans-serif, system-ui;
    background-color: var(--primary-bg) !important;
    color: var(--text-light);
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stBottomBlockContainer"] {
    background-color: var(--primary-bg);
}
[data-testid="stSidebar"], [data-testid="stHeader"], #MainMenu, footer {
    display: none;
}

/* --- PANTALLA DE API KEY --- */
div[data-testid="stForm"] button {
    background-color: var(--accent-green-dark);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    width: 100%;
    transition: background-color 0.3s ease;
}
div[data-testid="stForm"] button:hover {
    background-color: var(--accent-green);
    color: var(--primary-bg);
    border: none;
}
.stTextInput label {
    color: var(--text-light) !important;
}

/* --- INTERFAZ DE CHAT --- */
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
.st-emotion-cache-1c7y2kd { /* Nombre del rol (User/Assistant) */
    font-weight: 600; padding-bottom: 8px;
}

/* Estilos para las burbujas de chat */
[data-testid="stChatMessage"] > div[data-testid="stMarkdown"] {
    padding: 1.2em; border-radius: 12px; line-height: 1.6;
}
/* Respuestas del Agente: Fondo oscuro, texto blanco */
[data-testid="stChatMessage"][data-testid="chat-message-assistant"] > div[data-testid="stMarkdown"] {
    background-color: var(--secondary-bg);
    border: 1px solid var(--border-color);
    color: var(--text-light);
}
[data-testid="stChatMessage"][data-testid="chat-message-assistant"] .st-emotion-cache-1c7y2kd {
    color: var(--text-light);
}
/* Mensajes del Usuario: Fondo claro, texto negro */
[data-testid="stChatMessage"][data-testid="chat-message-user"] > div[data-testid="stMarkdown"] {
    background-color: var(--user-bg);
    color: var(--text-dark);
}
[data-testid="stChatMessage"][data-testid="chat-message-user"] .st-emotion-cache-1c7y2kd {
    color: var(--text-light);
}

/* Área de entrada de texto */
[data-testid="stChatInput"] {
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    background-color: rgba(43, 43, 43, 0.9); backdrop-filter: blur(10px);
    border-radius: 24px; padding: 8px 12px; border: 1px solid var(--border-color);
    width: 90%; max-width: 768px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
[data-testid="stChatInput"] textarea {
    background: none; color: var(--text-light);
}

/* <<--- ESTILOS PARA LOS BOTONES DE SUGERENCIA --->> */
.suggestion-buttons {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 10px;
    max-width: 768px;
    margin: 1rem auto 3rem auto;
}
.suggestion-buttons .stButton button {
    width: 100%;
    text-align: left;
    background-color: var(--secondary-bg);
    border: 1px solid var(--border-color);
    color: var(--text-light);
    padding: 12px 16px;
    border-radius: 12px;
    transition: background-color 0.2s ease, border-color 0.2s ease;
    font-size: 0.95em;
    font-weight: 500;
}
.suggestion-buttons .stButton button:hover {
    background-color: #3a3a3a;
    border-color: var(--accent-green);
}
</style>
""", unsafe_allow_html=True)


# --- LÓGICA DEL AGENTE CON MEMORIA ---
@st.cache_resource
def get_conversation_chain(_api_key):
    PROMPT = ChatPromptTemplate.from_messages([
        ("system", """Eres 'Bio Gemini', un agente de IA experto en biología. Responde de forma precisa, educativa y amigable. Utiliza el historial de conversación para entender el contexto de las nuevas preguntas.
        Reglas:
        - No des consejos médicos o veterinarios.
        - Usa **negritas** para términos clave."""),
        ("human", "{history}"),
        ("human", "{input}")
    ])
    llm = ChatGroq(api_key=_api_key, model="llama3-70b-8192")
    memory = ConversationBufferMemory(memory_key="history", input_key="input")
    conversation_chain = ConversationChain(llm=llm, prompt=PROMPT, verbose=False, memory=memory)
    return conversation_chain

# --- GESTIÓN DE ESTADO DE SESIÓN ---
if "groq_api_key" not in st.session_state: st.session_state.groq_api_key = None
if "messages" not in st.session_state: st.session_state.messages = []
if "chain" not in st.session_state: st.session_state.chain = None
if "selected_prompt" not in st.session_state: st.session_state.selected_prompt = None

# --- PANTALLA DE BIENVENIDA / API KEY ---
if not st.session_state.groq_api_key:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if logo_exists:
            img_col1, img_col2, img_col3 = st.columns([1,1,1])
            with img_col2:
                st.image(logo_path, width=80)
        else:
            st.markdown("<p style='font-size: 60px; text-align: center;'>🧬</p>", unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center; color: #FFFFFF;'>Bienvenido a Bio Gemini</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; opacity: 0.8;'>Para comenzar, ingresa tu API Key de Groq.</p>", unsafe_allow_html=True)
        
        with st.form("api_key_form"):
            api_key_input = st.text_input(
                "Tu API Key de Groq", type="password", placeholder="gsk_xxxxxxxxxx", 
                help="Obtén tu clave gratuita en console.groq.com", label_visibility="collapsed"
            )
            submitted = st.form_submit_button("Activar Bio Gemini")
            if submitted:
                if api_key_input and api_key_input.startswith("gsk_"):
                    st.session_state.groq_api_key = api_key_input
                    st.session_state.chain = get_conversation_chain(st.session_state.groq_api_key)
                    st.rerun()
                else:
                    st.error("Por favor, ingresa una API Key de Groq válida.")

# --- INTERFAZ DE CHAT PRINCIPAL ---
else:
    # --- TÍTULO Y BOTÓN DE NUEVO CHAT ---
    col1, col2, col3 = st.columns([2,3,2])
    with col1:
        st.markdown('<div style="text-align: left; padding-top: 1rem;">', unsafe_allow_html=True)
        if st.button("➕ Nuevo Chat"):
            st.session_state.messages = []
            if st.session_state.chain:
                st.session_state.chain.memory.clear()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='text-align: center; color: var(--accent-green);'>Bio Gemini</h1>", unsafe_allow_html=True)

    # --- LÓGICA DEL CHAT ---
    USER_AVATAR = "👤"
    BOT_AVATAR = logo_path if logo_exists else "✨"

    # <<--- LÓGICA PARA PREGUNTAS SUGERIDAS --->>
    if not st.session_state.messages:
        st.markdown("<h2 style='text-align: center; color: var(--text-light);'>¿Cómo puedo ayudarte hoy?</h2>", unsafe_allow_html=True)
        st.markdown("<div class='suggestion-buttons'>", unsafe_allow_html=True)
        cols = st.columns(2)
        suggestions = [
            "Explícame la fotosíntesis como si tuviera 10 años",
            "¿Qué es la edición genética con CRISPR-Cas9?",
            "Describe un animal que vive en las fosas abisales",
            "¿Cuáles son las diferencias clave entre mitosis y meiosis?"
        ]
        
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    st.session_state.selected_prompt = suggestion
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Mostrar historial de chat solo si la conversación ya ha comenzado
        for message in st.session_state.messages:
            avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    # Capturar la entrada del usuario (desde el input o desde los botones de sugerencia)
    prompt = st.chat_input("Pregúntame algo de biología...") or st.session_state.get("selected_prompt")

    if prompt:
        # Limpiar el prompt seleccionado para que no se use de nuevo
        if st.session_state.selected_prompt:
            st.session_state.selected_prompt = None
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            with st.spinner("Pensando..."):
                try:
                    response = st.session_state.chain.predict(input=prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = "Lo siento, ha ocurrido un error. Verifica tu API Key y tu conexión."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
        
        # Recargar la página para limpiar el prompt del input y mostrar la nueva respuesta
        st.rerun()
