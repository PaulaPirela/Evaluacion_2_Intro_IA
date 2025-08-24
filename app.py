import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import base64

# --- CONFIGURACIÓN DE LA PÁGINA ---
# Logo de Gemini en formato Base64 para usarlo como page_icon. 
# Este es el método más robusto para que Streamlit encuentre la imagen.
LOGO_IMAGE_BASE64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAFGklEQVRYw6WWa3BU1RXH/2/ubt6bN5PZbJJsJjebJBESSCgKlKgUqCgU+QcKx1IqLV2lY6VjpaV1tBoq2rGjY6V1tB2tY6VjqR2hUikgKEEQCAl5kBwS3mxy895t7t19zh/aJCSEmPj+zJ2dO/f8z/nO/5z/OWfG/4x/a4tQe+ORvBGt/bZJ29h5e7g/D9/8/3wI/bX78+eLzWfW30WwB49dF/eA24d/t87t3/z3/wB/gG/z3/gH/A//gP/gP/wH/gH/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/gP/g-AAAABJRU5ErkJggg=="

st.set_page_config(
    page_title="Bio Gemini",
    page_icon=LOGO_IMAGE_BASE64,
    layout="wide", # Usar "wide" para que el título centrado tenga más espacio
    layout="centered",
    initial_sidebar_state="auto"
)

# --- INYECCIÓN DE CSS PARA EMULAR LA INTERFAZ DE GEMINI ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

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
# Título centrado usando HTML
st.markdown("<h1 style='text-align: center; color: #e3e3e3;'>Bio Gemini</h1>", unsafe_allow_html=True)

# --- BARRA LATERAL PARA LA API KEY ---
with st.sidebar:
    st.header("🔑 Configuración")
    # Usamos un placeholder y ocultamos el label para un look más limpio
    groq_api_key = st.text_input(
        "Ingresa tu API Key de Groq:", 
        type="password", 
        key="groq_api_key_input", 
        label_visibility="collapsed", 
        placeholder="Ingresa tu API Key de Groq..."
    )
    
    if not groq_api_key:
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except (KeyError, FileNotFoundError):
            groq_api_key = ""

    st.markdown("---")
    st.info("Obtén tu API Key gratuita en [GroqCloud](https://console.groq.com/keys).")

# --- LÓGICA DEL AGENTE BIÓLOGO (ACTUALIZADO) ---
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """Eres 'Bio Gemini', un agente de IA experto en biología. Tu propósito es dar respuestas precisas y educativas.
         
         Tus capacidades principales son:
         1.  **Explicar Conceptos Biológicos**: Define y explica términos y conceptos (ej: '¿Qué es la meiosis?'). Usa analogías simples para temas complejos.
         2.  **Identificar Especies**: Si un usuario te da una descripción textual de un animal, planta, hongo o microorganismo, intenta identificar la especie. Siempre indica el nivel de confianza.
         3.  **Detallar Procesos Biológicos**: Explica procesos complejos paso a paso (ej: 'Explica la fotosíntesis').
         
         Reglas de Interacción:
         -   **Tono**: Mantén un tono didáctico, científico y amigable.
         -   **Precisión**: Prioriza la exactitud científica. Si no estás seguro de una respuesta, indícalo.
         -   **Seguridad**: No proporciones consejos médicos o veterinarios; recomienda consultar a un profesional.
         -   **Formato**: Usa **negritas** para resaltar términos clave y listas para organizar la información cuando sea apropiado."""),
        ("human", "{user_question}")
    ]
)

def get_chatbot_chain(api_key):
    llm = ChatGroq(api_key=api_key, model="llama3-70b-8192")
    return prompt_template | llm | StrOutputParser()

# --- INTERFAZ DE USUARIO PRINCIPAL ---
if groq_api_key:
    # Inicialización del historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hola, soy Bio Gemini. ¿Qué tema biológico exploramos hoy?"}]

    # Mostrar mensajes previos
    for message in st.session_state.messages:
        avatar_icon = "✨" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("Pregúntame algo de biología...", key="chat_input_main"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Pensando..."):
                chain = get_chatbot_chain(groq_api_key)
                try:
                    response = chain.invoke({"user_question": prompt})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error al contactar el modelo: {e}")
                    st.info("Verifica que tu API Key sea correcta y tenga saldo.")
else:
    st.warning("Por favor, ingresa tu API Key de Groq en la barra lateral para activar Bio Gemini.")
