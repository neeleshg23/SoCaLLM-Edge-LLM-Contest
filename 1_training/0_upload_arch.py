#%%
import torch
from transformers import JambaConfig, JambaForCausalLM, AutoTokenizer
from transformers.models.jamba.modeling_jamba import HybridMambaAttentionDynamicCache
from torchsummary import summary

# Load tokenizer from Llama 3.1
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")  # Replace with actual Llama 3.1 path
#%%
# Define Jamba model configuration
# Define Jamba model configuration
config = JambaConfig(
    vocab_size=len(tokenizer),
    d_model=1024,
    num_hidden_layers=3,
    num_attention_heads=16,  # Adjust number of attention heads as needed
    max_position_embeddings=2048,  # Example max length
    num_experts=4,
    tie_word_embeddings=True,
)

# Initialize the Jamba model
model = JambaForCausalLM(config)

# Calculate total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params / 1e6:.2f} million")
#%%
# model.save_pretrained('jamba-1.9b-3')
# tokenizer.save_pretrained('jamba-1.9b-3')

model.push_to_hub('neeleshg23/jamba-1.9b-3')
tokenizer.push_to_hub('neeleshg23/jamba-1.9b-3')
