import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
from transformers.models.jamba.modeling_jamba import HybridMambaAttentionDynamicCache
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

torch.set_num_threads(1)

device = torch.device("cpu")

# Load the dataset
dataset = load_dataset(
    'parquet',
    data_files={'train': '/home/neelesh/03_c4_heuristic_filtering/hf.parquet/part.0.parquet'}
)

dataset = dataset['train']

# Load model for CPU
model = AutoModelForCausalLM.from_pretrained(
    "neeleshg23/jamba-1.9b-3",
    torch_dtype=torch.float32,
    use_mamba_kernels=False
)
model = model.to(device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize function that creates labels for causal LM
def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=2048,
        return_tensors="pt"
    )
    
    # Create labels identical to input_ids for causal LM training
    result["labels"] = result["input_ids"].clone()
    
    return result

# Tokenize with proper labels
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    # num_proc=128
)

# Filter out empty entries
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)

# Use data collator specifically for language modeling
data_collator = default_data_collator

# Training arguments
output_dir = "./jamba_model_output"
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=5000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    dataloader_num_workers=0,
    gradient_accumulation_steps=1,
    fp16=False,
    report_to="none",
    use_cpu=True,
    # Add learning rate
    learning_rate=2e-5,
    # Enable proper loss calculation
    prediction_loss_only=True,
)

# Initialize Trainer with data collator
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# Resume training if checkpoint exists
last_checkpoint = get_last_checkpoint(output_dir) if os.path.exists(output_dir) else None
if last_checkpoint is not None:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

# Save the model and tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)