import torch
import torch.nn.functional as F
import gradio as gr
import tiktoken
from train import GPT, GPTConfig  # Import the model architecture from train.py

# Initialize model and tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPTConfig()
model = GPT(config).to(device)
model.load_state_dict(torch.load('model.pt', map_location=device))
model.eval()

# Initialize tokenizer
enc = tiktoken.get_encoding('gpt2')

def generate_text(prompt, max_length=30, temperature=0.8, top_k=50):
    # Encode the prompt
    input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0).to(device)
    
    # Generate text
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits from the model
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample from the distribution
            ix = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(top_k_indices, -1, ix)
            
            # Append to the sequence
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            # Stop if we generate an end of text token
            if next_token.item() == enc.eot_token:
                break
    
    # Decode the generated text
    generated_text = enc.decode(input_ids[0].tolist())
    return generated_text

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Enter your prompt", lines=2),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Text Generation with GPT",
    description="Enter a prompt and adjust the parameters to generate text using the trained GPT model."
)

# Launch the app
if __name__ == "__main__":
    iface.launch() 