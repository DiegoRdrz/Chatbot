from gpt4all import GPT4All
import gradio as gr

# Carga automática del modelo instruccionado
model = GPT4All("Meta-Llama-3--8BInstruct.Q4_0.gguf")

def generate_response(user_input, history=None):
    if history is None:
        history = []

    # Construir el prompt con todo el historial
    prompt = ""
    for (u, a) in history:
        prompt += f"User: {u}\nAssistant: {a}\n"
    prompt += f"User: {user_input}\nAssistant:"

    # Generar respuesta usando el parámetro `temp`
    out = model.generate(
        prompt,
        max_tokens=150,
        temp=0.7,        # ← parámetro correcto
        top_p=0.9,       # opcional
        top_k=40         # opcional
    )

    # Extraer texto de la salida
    response = (
        out["choices"][0]["text"].strip()
        if isinstance(out, dict) else out.strip()
    )

    # Actualizar el historial
    history.append((user_input, response))
    return history, history

# Interfaz web con Gradio
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Psiquiatra Virtual (Offline, Gratis)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="How are you feeling today?")
    clear = gr.Button("Clear")
    state = gr.State([])

    msg.submit(generate_response, [msg, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])

demo.launch()  # Abre http://localhost:7860
