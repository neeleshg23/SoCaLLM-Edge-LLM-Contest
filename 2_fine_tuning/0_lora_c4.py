import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained(
    "neeleshg23/jamba-1.9b-3",
    # device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

lora_config = LoraConfig(
    r=8,
    target_modules=[
        "embed_tokens",
        "x_proj", "in_proj", "out_proj", # mamba
        "gate_proj", "up_proj", "down_proj", # mlp
        "q_proj", "k_proj", "v_proj", "o_proj", # attention
    ],
    task_type="CAUSAL_LM",
    bias="none",
)

# dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")
dataset = load_dataset('parquet', data_files={'train': [f'/home/neelesh/03_c4_heuristic_filtering/hf.parquet/part.{i}.parquet' for i in range(100, 200)]})
training_args = SFTConfig(
    output_dir="./jamba_lora_model_output",
    logging_dir="./logs",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    logging_steps=10,
    gradient_checkpointing=True,
    max_seq_length=2048,
    save_steps=100,
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
)
trainer.train()