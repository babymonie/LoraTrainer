import random
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments, 
    TrainerCallback
)
from datasets import Dataset
import os
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
import fitz

# ----------------- SEED -----------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed_everything(42)

# ----------------- QUANT CONFIG -----------------
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ----------------- MODEL LOADING -----------------
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    token=True
)

# ----------------- MODEL PREPARATION -----------------
model = prepare_model_for_kbit_training(model)

# ----------------- PEFT CONFIG -----------------
lora_config = LoraConfig(
    r=128,  # Reduced from 128 to prevent overfitting
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ----------------- TRAINING ARGS -----------------
training_args = TrainingArguments(
    output_dir="./lora_outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=30,  # Reduced epochs
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    optim="paged_adamw_32bit",
    save_strategy="no",
    report_to="none"
)

# ----------------- DATA LOADING -----------------
def load_files(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        content = ""
        
        if file.endswith(".txt"):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        elif file.endswith(".pdf"):
            doc = fitz.open(path)
            content = "\n".join([page.get_text().strip() for page in doc])
        
        if content:
            documents.append({"text": content})
            
    return documents
uploaded_data_folder = "uploaded_docs"
if not os.path.exists(uploaded_data_folder):
    raise FileNotFoundError(f"Directory {uploaded_data_folder} does not exist.")


# ----------------- TRAINING SETUP -----------------
trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    args=training_args,
    train_dataset=Dataset.from_list(load_files("uploaded_docs"))
)

# ----------------- TRAINING EXECUTION -----------------
# ----------------- TRAINING EXECUTION -----------------
try:
    trainer.train()
    
    # CORRECTED SAVING
    trainer.model.save_pretrained("./lora_adapter")
    tokenizer.save_pretrained("./lora_adapter")
    
    print("Training completed successfully.")
    print("LoRA adapter saved to ./lora_adapter.")
except Exception as e:
    print(f"Training failed: {str(e)}")
    raise