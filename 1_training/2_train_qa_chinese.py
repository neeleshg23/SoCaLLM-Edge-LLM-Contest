import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

torch.set_num_threads(1)

# Load the dataset
dataset = load_dataset("silk-road/alpaca-data-gpt4-chinese")

# Define formatting for question-answer prompts
def format_prompt(example):
    """
    根据是否包含输入字段生成新的提示格式。
    """
    if example.get("input"):
        # 有输入的情况
        prompt = ("下面是一条描述任务的指令，并配有提供进一步上下文的输入。"
                  "请写出一个适当的回答来完成该请求。\n\n"
                  "### 指令:\n{instruction}\n\n"
                  "### 输入:\n{input}\n\n"
                  "### 回答:\n").format(
            instruction=example["instruction"],
            input=example["input"]
        )
    else:
        # 没有输入的情况
        prompt = ("下面是一条描述任务的指令。"
                  "请写出一个适当的回答来完成该请求。\n\n"
                  "### 指令:\n{instruction}\n\n"
                  "### 回答:\n").format(
            instruction=example["instruction"]
        )
    return {"text": prompt}

# Apply formatting to dataset
formatted_dataset = dataset["train"].map(
    format_prompt,
    remove_columns=dataset["train"].column_names,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        padding=True,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )
    # Create labels identical to input_ids for causal LM training
    result["labels"] = result["input_ids"].clone()
    
    # Log shapes to identify any empty tensors
    if result["input_ids"].shape[0] == 0:
        print("Empty tensor found in input_ids!")
    return result

# Remove unwanted columns and tokenize
tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=32, 
    remove_columns=formatted_dataset.column_names,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
    use_mamba_kernels=True,
)

# Training arguments
output_dir = "./chinese_alpaca_output"
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=5000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    dataloader_num_workers=4,
    fp16=False,
    report_to="none",
    learning_rate=5e-5,
    prediction_loss_only=True,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    push_to_hub=True,
    ddp_find_unused_parameters=False,
    hub_strategy="every_save",
    hub_model_id="your_hub_model_id",
    torch_compile=True,
    weight_decay=0.01,
    max_grad_norm=1.0,
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