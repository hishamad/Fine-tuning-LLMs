import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Model Configuration
model_name = "SWAH-KTH/large_lora_model_q"
model_file = "unsloth.Q4_K_M.gguf"
model_path = hf_hub_download(model_name, filename=model_file)

llm = Llama(
    model_path=model_path,
    n_threads=32,  # Adjust threads for your CPU
    # n_gpu_layers=0  # Uncomment if using GPU
)

def respond(message, history, scenario, max_tokens, temperature, top_p):
    """
    Respond to the user's message.
    """
    # Build the dynamic system message with the chosen scenario
    system_message = f"You are acting as a {scenario} role in a scenario. Only answer as {scenario} role, never include anything as if you were an User role. Just fill in the assistant answer."
    full_prompt = system_message + "\n"
    print(system_message)
    # Include the chat history in the prompt
    for user_msg, assistant_msg in history:
        full_prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
    full_prompt += f"User: {message}\nAssistant: "

    response = ""
    try:
        res = llm(full_prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
        response = res["choices"][0]["text"] if isinstance(res, dict) else res
    except Exception as e:
        response = f"Error: {str(e)}"
    
    # Update the history
    history.append((message, response))
    return history, history

def clear_chat():
    """
    Clear the conversation history.
    """
    return [], []

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎭 Role-Playing Chatbot")
    gr.Markdown("Select a scenario and engage with the chatbot!")

    with gr.Row():
        with gr.Column(scale=1):
            scenario = gr.Dropdown(
                choices=["Job Interviewer", "Salesperson", "Language Tutor", "Debate Opponent"],
                label="Choose a Scenario",
                value="Job Interviewer",
            )
            max_tokens = gr.Slider(
                minimum=1, maximum=512, value=128, step=1, label="Max Response Tokens"
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"
            )
            top_p = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p Sampling"
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", elem_id="chatbot")
            user_input = gr.Textbox(
                placeholder="Type your message here...", label="Your Message"
            )
            with gr.Row():
                send_button = gr.Button("Send")
                clear_button = gr.Button("Clear Chat")

    history = gr.State([])

    send_button.click(
        respond,
        inputs=[user_input, history, scenario, max_tokens, temperature, top_p],
        outputs=[chatbot, history],
    )
    clear_button.click(clear_chat, inputs=[], outputs=[chatbot, history])

if __name__ == "__main__":
    demo.launch()
