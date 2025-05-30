import os
import json
from gpt4all import GPT4All
import gradio as gr
import random


from analyzer import analyze_text

# Cargar modelo con manejo de errores
try:
    model = GPT4All(
        model_name="Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",
        device="cuda",
        verbose=False
    )
except Exception as e:
    print(f"Error al cargar el modelo en GPU: {e}. Cambiando a CPU...")
    model = GPT4All(
        model_name="Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",
        device="cpu",
        verbose=False
    )

# Directorio para guardar historial de sesiones
HISTORY_DIR = "session_memory"
os.makedirs(HISTORY_DIR, exist_ok=True)

ANALYSIS_DIR = "session_analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)


# Optimized instruction prompt for a psychiatrist chatbot
INSTRUCTION = (
    "You are a licensed, experienced psychiatrist conducting a one-on-one therapy session with a patient.\n"
    "You are speaking directly to the patient, and your only role is that of the psychiatrist. \n"
    "You are warm, understanding, and deeply empathetic. \n"
    "Your tone is natural, emotionally attuned, calm, and human — never robotic or scripted.\n"
    "\n"
    "Your job is to:\n"
    "- Listen attentively to what the patient says.\n"
    "- Respond with supportive, validating, and compassionate replies.\n"
    "- Gently explore the patient’s thoughts, emotions, and experiences.\n"
    "- Ask open-ended questions that encourage reflection and self-awareness.\n"
    "- Show no judgment and never rush the patient.\n"
    "- Maintain emotional sensitivity to distress, sadness, confusion, anger, anxiety, or trauma.\n"
    "- Provide a safe, affirming space for the patient to share.\n"
    "\n"
    "You must:\n"
    "- Never include instructions like 'Please respond', 'Go ahead', or '(Your response)'.\n"
    "- Never explain your role, the setting, or anything meta about the conversation.\n"
    "- Never use 'Patient:' or 'Psychiatrist:' inside your responses.\n"
    "- Never break character or mention that this is a simulation.\n"
    "- Never repeat the same phrases or questions.\n"
    "- Never talk about this being a chatbot, AI, assistant, or anything technical.\n"
    "\n"
    "Instead, always reply as if you're sitting face-to-face with a real person in a private, safe space.\n"
    "Begin every response naturally — as if continuing the conversation — and maintain emotional continuity.\n"
    "\n"
    "Examples of good follow-up techniques include:\n"
    "- 'Can you tell me more about that?'\n"
    "- 'How did that make you feel?'\n"
    "- 'What was going through your mind at the time?'\n"
    "- 'Why do you think that affected you so deeply?'\n"
    "- 'How have you been coping with this?'\n"
    "- 'Have these feelings changed over time?'\n"
    "\n"
    "If the patient expresses intense emotion (like sadness, guilt, hopelessness), respond with validation and calm reassurance.\n"
    "Do not attempt to fix, diagnose, or correct. Your goal is to **hold space**, not to solve.\n"
    "\n"
    "You are not a narrator or scriptwriter. Just reply as a psychiatrist in session — one message at a time.\n"
    "\n"
    "Remember: no instructions, no commentary, no roles. Just authentic, supportive replies.\n"
)


# Función para limpiar la respuesta del modelo
def clean_response(response):
    response = response.replace("Please respond with empathy and support.", "").strip()
    return response

# Función para guardar el historial de conversación
def save_history(session_id, history):
    filepath = os.path.join(HISTORY_DIR, f"{session_id}.json")
    with open(filepath, "w") as f:
        json.dump(history, f)

# Función para cargar el historial de conversación
def load_history(session_id):
    filepath = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []

def save_analysis(session_id, result):
    filepath = os.path.join(ANALYSIS_DIR, f"{session_id}.json")
    data = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
    data.append(result)
    with open(filepath, "w") as f:
        json.dump(data, f)

def load_analysis(session_id):
    filepath = os.path.join(ANALYSIS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []

# Función para formatear el historial de conversación
def format_conversation(history, user_input):
    formatted_history = "\n".join(
        [f"Patient: {entry[0]}\nPsychiatrist: {entry[1]}" for entry in history]
    )
    prompt = (
        f"{INSTRUCTION}\n\n"
        f"Conversation history (for context only):\n"
        f"{formatted_history}\n\n"
        f"New interaction:\n"
        f"Patient: {user_input}\nPsychiatrist:"
    )
    return prompt

# Función para generar la respuesta del modelo
def generate_response(user_input, session_id):
    history = load_history(session_id)
    prompt = format_conversation(history, user_input)

    # Generar respuesta
    out = model.generate(
        prompt,
        max_tokens=300,
        temp=0.75,
        top_p=0.95,
        top_k=40
    )

    response = out.strip() if isinstance(out, str) else out["choices"][0]["text"].strip()
    response = clean_response(response)

    history.append((user_input, response))
    save_history(session_id, history)

    analysis_result = analyze_text(user_input)
    save_analysis(session_id, analysis_result)
    return history, history

# Función para reiniciar la sesión
def reset_session(session_id):
    filepath = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
    # Eliminar análisis asociado a la sesión
    filepath = os.path.join(ANALYSIS_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
    return [], []

# Finalizar sesion
def finish_session(session_id):
    if not session_id:
        return "Please enter a Session ID."
    
    analysis = load_analysis(session_id)
    if not analysis:
        return "No analysis found for this session."

    active_messages = [
        a for a in analysis if (
            a.get('suicidal_phrase') or
            a.get('sentiment') in ['positive', 'negative'] or
            a.get('emotion') in ['joy', 'trust', 'sadness', 'fear', 'anger']
        )
    ]   
    total = len(active_messages)
    
    if total == 0:
        return "No messages found in this session."

    # Contadores mejorados (evitando doble conteo por mensaje)
    suicide_flags = 0
    negative_signals = 0
    positive_signals = 0
    
    for a in analysis:
        # Contar suicidio (máximo 1 por mensaje)
        if a.get('suicidal_phrase'):
            suicide_flags += 1
        
        # Señales negativas (máximo 1 por mensaje)
        if (a.get('sentiment') == 'negative' or 
            a.get('emotion') in ['sadness', 'fear', 'anger'] or 
            a.get('intent') == 'express_negative_emotion'):
            negative_signals += 1
        
        # Señales positivas (máximo 1 por mensaje)
        if (a.get('sentiment') == 'positive' or 
            a.get('emotion') in ['joy', 'trust'] or 
            a.get('intent') == 'express_positive_emotion'):
            positive_signals += 1

    # Pesos
    weight_suicide = 10 * suicide_flags
    weight_negative = 1.0
    weight_positive = -0.25

    # Cálculo mejorado con máximos realistas
    raw_score = (
        (suicide_flags * weight_suicide) +
        (negative_signals * weight_negative) +
        (positive_signals * weight_positive)
    )
    
    # Máximo teórico realista (todos los mensajes son suicidas + negativos)
    max_possible = (weight_suicide + weight_negative) * total
    # Mínimo teórico (todos los mensajes son p  ositivos)
    min_possible = weight_positive * total
    
    # Normalización ajustada
    if max_possible - min_possible <= 0:
        probability = 0.0
    else:
        normalized = (raw_score - min_possible) / (max_possible - min_possible)
        probability = round(max(0, min(normalized, 1)) * 100)

    return f"Estimated probability of depression: {probability}%\n(based on {total} user messages)"

# Lista para almacenar sesiones activas
active_sessions = set()

def generate_unique_session_id():
    while True:
        session_id = f"session_{random.randint(1000, 9999)}"
        if session_id not in active_sessions:
            active_sessions.add(session_id)
            return session_id

# Interfaz gráfica con Gradio
with gr.Blocks(css="body {background-color: #f9fafb;} .gr-button {font-size: 16px;}") as demo:
    gr.Markdown("<h1 style='text-align: center; color: #333;'>PsycoBot: An AI Mental Health Assistant</h1>")

    with gr.Row():
        session_id = gr.Label(label="Session ID")  # visible por defecto

    history = gr.State([])

    with gr.Row():
        chatbot = gr.Chatbot(label="Conversation", height=400)
    
    with gr.Row():
        msg = gr.Textbox(label="Your message", placeholder="Write here and press Enter", lines=1)

    with gr.Row():
        clear = gr.Button("Clear Session")
        finish = gr.Button("Finish Session")

    result_output = gr.Textbox(label="Depression Risk (after finishing)", interactive=False)

    # Función de respuesta
    def respond(user_input, session_id, history):
        if not session_id:
            return gr.update(), history
        history, updated_history = generate_response(user_input, session_id)
        return updated_history, ""

    # Función para limpiar sesión
    def clear_session(session_id):
        if session_id:
            reset_session(session_id)
        return [], ""

    # Función para inicializar sesión al cargar
    def initialize_session():
        sid = generate_unique_session_id()
        return sid, []

    # Conexiones
    msg.submit(respond, inputs=[msg, session_id, history], outputs=[chatbot, msg])
    clear.click(clear_session, inputs=[session_id], outputs=[chatbot, msg])
    finish.click(finish_session, inputs=[session_id], outputs=[result_output])

    # Inicializar valores al cargar
    demo.load(fn=initialize_session, inputs=[], outputs=[session_id, history])

# Ejecutar la aplicación
if __name__ == "__main__":
    demo.launch(share=True)
