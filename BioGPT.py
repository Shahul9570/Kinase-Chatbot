import torch
import gradio as gr
import time
import threading
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and BioGPT model from local checkpoint
checkpoint_dir = r"C:\Users\ASUS\Desktop\CIODS 1\New folder\content\results\content\results\checkpoint-final-full"
try:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if not hasattr(config, "model_type") or config.model_type is None:
        config.model_type = "biogpt"  # Explicitly set if missing
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, config=config).to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

stop_event = threading.Event()


def generate_response(prompt):
    global stop_event
    stop_event.clear()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    animated_response = ""
    for char in response:
        if stop_event.is_set():
            return "Generation stopped."
        animated_response += char
        time.sleep(0.05)
        yield animated_response


def stop_generation():
    global stop_event
    stop_event.set()
    return "Generation stopped. Please enter a new prompt."

# Gradio interface
def create_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            prompt_input = gr.Textbox(label="Enter a prompt", placeholder="Type a prompt...")
            stop_button = gr.Button("Stop")
        with gr.Row():
            output_textbox = gr.Textbox(label="Response", interactive=False)
        
        # Fix: No gr.State usage
        stop_button.click(fn=stop_generation, inputs=[], outputs=output_textbox)
        prompt_input.submit(fn=generate_response, inputs=prompt_input, outputs=output_textbox)
    
    return demo

iface = create_interface()
iface.launch()

