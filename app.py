import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Bio Agent",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- INYECCIÓN DE CSS PARA EMULAR LA INTERFAZ DE GEMINI ---
st.markdown("""
<style>
/* Reset básico y configuración de fuente */
body {
    font-family: 'Google Sans', sans-serif, system-ui;
    background-color: #131314; /* Fondo principal de Gemini */
    color: #e3e3e3;
}

/* Contenedor principal de la aplicación */
.stApp {
    background-color: #131314;
}

/* Barra lateral */
[data-testid="stSidebar"] {
    background-color: #1e1f20;
    border-right: 1px solid #3c4043;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
    color: #e3e3e3;
}

/* Área de entrada del chat (Prompt) */
[data-testid="stChatInput"] {
    background-color: #1e1f20;
    border-radius: 28px; /* Bordes redondeados */
    padding: 10px 15px;
    border: 1px solid #3c4043;
    margin: 1rem auto;
    max-width: 768px; /* Ancho similar al de Gemini */
}
[data-testid="stChatInput"] textarea {
    background: none;
    color: #e3e3e3;
}
[data-testid="stChatInput"] button {
    border-radius: 50%;
}

/* Contenedor de los mensajes del chat */
[data-testid="stChatMessage"] {
    background: none; /* Sin fondo en las burbujas */
    padding: 0;
    margin: 2rem 0;
    border-radius: 0;
}
.stChatMessage {
    max-width: 768px;
    margin: 0 auto;
}

/* Estilo para el contenido del mensaje (el texto en sí) */
[data-testid="stChatMessage"] > div[data-testid="stMarkdown"] {
    padding: 1em;
    border-radius: 12px;
}
/* Estilo específico para el mensaje del asistente (IA) */
[data-testid="stChatMessage"][data-testid="chat-message-assistant"] > div[data-testid="stMarkdown"] {
    background-color: #1e1f20; /* Fondo sutil para la respuesta */
}

/* Ocultar elementos de Streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🧬 Bio Agent")
# Se elimina el caption para un look más limpio

# --- BARRA LATERAL PARA LA API KEY ---
with st.sidebar:
    st.header("🔑 Configuración")
    groq_api_key = st.text_input("Ingresa tu API Key de Groq:", type="password", key="groq_api_key_input", label_visibility="collapsed", placeholder="Ingresa tu API Key de Groq...")
    
    if not groq_api_key:
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except (KeyError, FileNotFoundError):
            groq_api_key = ""

    st.markdown("---")
    st.info("Obtén tu API Key gratuita en [GroqCloud](https://console.groq.com/keys).")

# --- LÓGICA DEL AGENTE BIÓLOGO (sin cambios) ---
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """Eres 'Bio Agent', un agente de IA experto en biología. Tu propósito es dar respuestas precisas y educativas.
         Tus capacidades:
         1.  **Explicar Conceptos**: Define y explica términos biológicos.
         2.  **Identificar Especies**: A partir de una descripción, intenta identificar la especie e indica tu nivel de confianza.
         3.  **Detallar Procesos**: Explica procesos complejos paso a paso.
         Reglas:
         -   **Tono**: Didáctico, científico y amigable.
         -   **Precisión**: Prioriza la exactitud. Si no estás seguro, indícalo.
         -   **Seguridad**: No des consejos médicos o veterinarios; recomienda consultar a un profesional.
         -   **Formato**: Usa **negritas** para términos clave y listas para organizar la información."""),
        ("human", "{user_question}")
    ]
)

def get_chatbot_chain(api_key):
    llm = ChatGroq(api_key=api_key, model="llama3-70b-8192")
    return prompt_template | llm | StrOutputParser()

# --- INTERFAZ DE USUARIO PRINCIPAL ---
if groq_api_key:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hola, soy Bio Agent. ¿Qué tema biológico exploramos hoy?"}]

    for message in st.session_state.messages:
        # Usamos los íconos 🧬 para el asistente y 👤 para el usuario
        avatar_icon = "🧬" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pregúntame algo de biología...", key="chat_input_main"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🧬"):
            with st.spinner("Pensando..."):
                chain = get_chatbot_chain(groq_api_key)
                try:
                    response = chain.invoke({"user_question": prompt})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error al contactar el modelo: {e}")

else:
    st.warning("Por favor, ingresa tu API Key de Groq en la barra lateral para comenzar.")
