from gpt4all import GPT4All
import gradio as gr

# Carga del modelo con aceleración por GPU
model = GPT4All(
    model_name="Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    device="cuda",
    verbose=True
)

# Prompt inicial que define el comportamiento del modelo como un psiquiatra empático
# Modificar la instrucción inicial
INSTRUCTION = (
    "You are a compassionate and supportive psychiatrist. "
    "Your role is to listen and respond empathetically to the patient's messages. "
    "Keep your replies emotionally supportive and focused on the patient's well-being.\n\n"
)

def generate_response(user_input, history=None):
    if history is None:
        history = []

    # Construir el historial de la conversación
    conversation = ""
    for (u, a) in history:
        conversation += f"Patient: {u}\nPsychiatrist: {a}\n"
    conversation += f"Patient: {user_input}\nPsychiatrist:"

    full_prompt = INSTRUCTION + conversation

    # Generar respuesta
    out = model.generate(
        full_prompt,
        max_tokens=250,
        temp=0.75,
        top_p=0.95,
        top_k=50
    )

    # Limpiar salida del modelo
    response = (
        out["choices"][0]["text"].strip()
        if isinstance(out, dict) else out.strip()
    )

    # Filtrar la respuesta para eliminar instrucciones no deseadas
    response = response.replace("Please respond as the psychiatrist.", "").strip()

    history.append((user_input, response))
    return history, history, ""


# Interfaz con Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Virtual Psychiatrist ")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="How are you feeling today?")
    clear = gr.Button("Clear")
    state = gr.State([])

    msg.submit(generate_response, [msg, state], [chatbot, state, msg])
    clear.click(lambda: ([], [], ""), None, [chatbot, state, msg])

demo.launch()
