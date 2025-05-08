from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
LORA_PATH = "./lora_adapter"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        token=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def load_lora_model():
    base_model, tokenizer = load_base_model()
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    lora_layers = [name for name, _ in model.named_modules() if "lora" in name]
    print(f"LoRA layers loaded: {len(lora_layers)}")  # Should be >0
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load models
base_model, base_tokenizer = load_base_model()
lora_model, lora_tokenizer = load_lora_model()

# Test prompt
prompt = "In the case of Smith v. Greenfield Corporation, why did the court rule in favor of Jane Smith despite Greenfield Corporation claiming the software was defective?"

print("=== BASE MODEL ===")
base_output = generate_response(base_model, base_tokenizer, prompt)
print(base_output)

print("\n=== LoRA MODEL ===")
lora_output = generate_response(lora_model, lora_tokenizer, prompt)
print(lora_output)
