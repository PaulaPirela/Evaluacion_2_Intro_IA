import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Bio-Asistente IA",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- TÍTULO Y DESCRIPCIÓN ---
st.title("🧬 Bio-Asistente IA")
st.caption("Tu experto en biología personal, impulsado por IA")
st.markdown("""
Bienvenido al Bio-Asistente. Soy un agente especializado en biología, entrenado para ayudarte con tus dudas. 
Puedes preguntarme sobre:
- **Conceptos biológicos**: Fotosíntesis, mitosis, genética, etc.
- **Identificación de especies**: Describe un animal o planta y trataré de identificarlo.
- **Procesos complejos**: Explícame el ciclo de Krebs, la replicación del ADN, etc.

Para comenzar, por favor ingresa tu API Key de Groq en la barra lateral.
""")

# --- BARRA LATERAL PARA LA API KEY ---
with st.sidebar:
    st.header("🔑 Configuración")
    # Usamos st.text_input con tipo 'password' para ocultar la clave
    groq_api_key = st.text_input("Ingresa tu API Key de Groq:", type="password", key="groq_api_key_input")
    
    # Intenta cargar la clave desde los secretos de Streamlit si no se ingresa manualmente
    if not groq_api_key:
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except (KeyError, FileNotFoundError):
            groq_api_key = "" # Mantener vacío si no se encuentra en ningún lado

    st.markdown("---")
    st.info("Obtén tu API Key gratuita en [GroqCloud](https://console.groq.com/keys).")
    st.info("Tu clave no será almacenada, solo se usa para esta sesión.")


# --- LÓGICA DEL AGENTE BIÓLOGO ---

# 1. Plantilla del Prompt (Instrucciones para el LLM)
# Esta es la parte más importante para especializar al agente.
# Le decimos cómo debe comportarse, qué rol debe adoptar y cómo debe responder.
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """Eres 'Bio-Asistente', un agente de inteligencia artificial experto en biología. Tu propósito es proporcionar respuestas precisas, claras y educativas sobre cualquier tema biológico.
         
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

# 2. Inicialización del Chatbot
def get_chatbot_chain(api_key):
    """Crea y devuelve la cadena de LangChain para el chatbot."""
    # Usamos el modelo Llama 3 de 70 mil millones de parámetros, que es excelente para razonamiento.
    llm = ChatGroq(
        api_key=api_key,
        model="llama3-70b-8192"
    )
    # Creamos la "cadena" que une el prompt, el modelo y el procesador de salida.
    return prompt_template | llm | StrOutputParser()

# --- INTERFAZ DE USUARIO PRINCIPAL ---

# Verificación de la API Key
if groq_api_key:
    st.success("API Key cargada correctamente. ¡Listo para recibir tus preguntas!")
    
    # Inicialización del historial de chat en el estado de la sesión
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes previos
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("Pregúntame algo sobre biología..."):
        # Añadir mensaje del usuario al historial y mostrarlo
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar y mostrar respuesta del asistente
        with st.chat_message("assistant"):
            with st.spinner("Pensando... 🧠"):
                chain = get_chatbot_chain(groq_api_key)
                try:
                    response = chain.invoke({"user_question": prompt})
                    st.markdown(response)
                    # Añadir respuesta del asistente al historial
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error al contactar el modelo: {e}")
                    st.info("Verifica que tu API Key sea correcta y tenga saldo.")

else:
    st.warning("Por favor, ingresa tu API Key de Groq en la barra lateral para activar el Bio-Asistente.")
    st.image("https://i.imgur.com/3g2Q1m9.png", caption="Ingresa tu API Key en el campo de la izquierda.", use_column_width=True)
