import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

torch.set_num_threads(1)

# Load the dataset
dataset = load_dataset(
    'parquet',
    data_files={'train': [f'/home/neelesh/03_c4_heuristic_filtering/hf.parquet/part.{i}.parquet' for i in range(100)]}
)

dataset = dataset['train']

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        padding=True,
        max_length=1024,
        truncation=True,  # Added truncation
        return_tensors="pt"
    )
    # Create labels identical to input_ids for causal LM training
    result["labels"] = result["input_ids"].clone()
    
    # Log shapes to identify any empty tensors
    if result["input_ids"].shape[0] == 0:
        print("Empty tensor found in input_ids!")
    return result

# Remove unwanted columns and tokenize
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=32, 
    remove_columns=dataset.column_names,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "neeleshg23/jamba-1.9b-3",
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
    use_mamba_kernels=True,
)

# Training arguments
output_dir = "./jamba_new_model_output"
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=14,
    save_steps=5000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    dataloader_num_workers=4,
    fp16=False,
    report_to="none",
    learning_rate=1e-4,
    prediction_loss_only=True,
    bf16=True,
    gradient_checkpointing=False,
    optim="adamw_torch_fused",
    push_to_hub=True,
    ddp_find_unused_parameters=False,
    # Hub pushing configuration
    hub_strategy="every_save",  # Pushes to hub at every save
    hub_model_id="neeleshg23/jamba-1.9b-3",
    torch_compile=True,
    weight_decay=0,
    max_grad_norm=None
)
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=default_data_collator,
    train_dataset=tokenized_dataset,
)

# Train
last_checkpoint = get_last_checkpoint(output_dir) if os.path.exists(output_dir) else None
if last_checkpoint is not None:
    print(f"Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

# Save
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)