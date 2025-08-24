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

# --- INYECCIÓN DE CSS PARA TEMA OSCURO (ESTILO GEMINI) ---
st.markdown("""
<style>
/* Colores principales */
body {
    color: #fafafa;
    background-color: #0e1117;
}

/* Color de fondo secundario (para la barra lateral y otros elementos) */
.stApp {
    background-color: #0e1117;
}

/* Estilo de los contenedores de chat */
[data-testid="stChatMessage"] {
    background-color: #262730;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Color del texto del input de la API Key */
[data-testid="stTextInput"] input {
    color: #fafafa;
}

/* Estilo de los botones */
.stButton>button {
    border-color: #8884d8;
    color: #8884d8;
}

.stButton>button:hover {
    border-color: #fafafa;
    color: #fafafa;
    background-color: #8884d8;
}

/* Ocultar el menú principal y el pie de página de Streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🧬 Bio Agent")
st.caption("Tu experto en biología personal, impulsado por IA")
st.markdown("""
Bienvenido a Bio Agent. Soy un agente especializado en biología, listo para ayudarte.
Puedes consultarme sobre:
- **Conceptos biológicos**: Fotosíntesis, mitosis, genética, etc.
- **Identificación de especies**: Describe un ser vivo y trataré de identificarlo.
- **Procesos complejos**: Explícame el ciclo de Krebs, la replicación del ADN, etc.

Para comenzar, por favor ingresa tu API Key de Groq en la barra lateral.
""")

# --- BARRA LATERAL PARA LA API KEY ---
with st.sidebar:
    st.header("🔑 Configuración")
    groq_api_key = st.text_input("Ingresa tu API Key de Groq:", type="password", key="groq_api_key_input")
    
    if not groq_api_key:
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except (KeyError, FileNotFoundError):
            groq_api_key = ""

    st.markdown("---")
    st.info("Obtén tu API Key gratuita en [GroqCloud](https://console.groq.com/keys).")
    st.info("Tu clave no será almacenada, solo se usa para esta sesión.")

# --- LÓGICA DEL AGENTE BIÓLOGO ---
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """Eres 'Bio Agent', un agente de inteligencia artificial experto en biología. Tu propósito es proporcionar respuestas precisas, claras y educativas sobre cualquier tema biológico.
         
         Tus capacidades principales son:
         1.  **Explicar Conceptos Biológicos**: Define y explica términos y conceptos (ej: '¿Qué es la meiosis?'). Usa analogías simples para temas complejos.
         2.  **Identificar Especies**: Si un usuario te da una descripción textual de un animal, planta, hongo o microorganismo (características físicas, hábitat, comportamiento), intenta identificar la especie. Siempre indica el nivel de confianza de tu identificación (ej: 'Basado en tu descripción, es muy probable que sea un...').
         3.  **Detallar Procesos Biológicos**: Explica procesos complejos paso a paso (ej: 'Explica la fotosíntesis'). Si es necesario, divide el proceso en fases claras.

         Reglas de Interacción:
         -   **Tono**: Mantén un tono amigable, didáctico y científico.
         -   **Precisión**: Prioriza la exactitud científica. Si no estás seguro de una respuesta, indícalo.
         -   **Claridad**: Evita la jerga excesiva. Si usas un término técnico, explícalo brevemente.
         -   **Seguridad**: No proporciones consejos médicos o veterinarios. Si la pregunta se relaciona con la salud humana o animal, recomienda consultar a un profesional.
         -   **Formato**: Utiliza negritas para resaltar términos clave y listas para organizar la información cuando sea apropiado."""),
        ("human", "{user_question}")
    ]
)

def get_chatbot_chain(api_key):
    llm = ChatGroq(
        api_key=api_key,
        model="llama3-70b-8192"
    )
    return prompt_template | llm | StrOutputParser()

# --- INTERFAZ DE USUARIO PRINCIPAL ---
if groq_api_key:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hola, ¿en qué puedo ayudarte hoy?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pregúntame algo sobre biología..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando... 🧠"):
                chain = get_chatbot_chain(groq_api_key)
                try:
                    response = chain.invoke({"user_question": prompt})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error al contactar el modelo: {e}")
                    st.info("Verifica que tu API Key sea correcta y tenga saldo.")
else:
    st.warning("Por favor, ingresa tu API Key de Groq en la barra lateral para activar Bio Agent.")
    st.info("La interfaz de chat aparecerá aquí una vez que la clave sea validada.")
