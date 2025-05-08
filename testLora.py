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

def load_lora_model():
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        token=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    # Verify LoRA application
    lora_layers = [name for name, _ in model.named_modules() if "lora" in name]
    print(f"LoRA layers loaded: {len(lora_layers)}")  # Should be >0
    
    return model, AutoTokenizer.from_pretrained(MODEL_NAME)

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Test
lora_model, lora_tokenizer = load_lora_model()
generate_response(lora_model, lora_tokenizer, prompt = "In the case of Smith v. Greenfield Corporation, why did the court rule in favor of Jane Smith despite Greenfield Corporation claiming the software was defective?")