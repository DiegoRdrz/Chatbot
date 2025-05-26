import os
import json
from gpt4all import GPT4All
import gradio as gr

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
    return history, history

# Función para reiniciar la sesión
def reset_session(session_id):
    filepath = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
    return [], []

# Interfaz gráfica con Gradio
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message")
    session_id = gr.Textbox(label="Session ID", placeholder="Enter a session ID (optional)")
    clear = gr.Button("Clear")

    history = gr.State([])

    def respond(user_input, session_id, history):
        if session_id:
            history = load_history(session_id)
        history, updated_history = generate_response(user_input, session_id)
        return updated_history, updated_history

    def clear_session(session_id):
        if session_id:
            reset_session(session_id)
        return [], []

    msg.submit(respond, inputs=[msg, session_id, history], outputs=[chatbot, history])
    clear.click(clear_session, inputs=[session_id], outputs=[chatbot, history])

# Ejecutar la aplicación
if __name__ == "__main__":
    demo.launch()
