

# LoRA Enhanced Text Generation

This project demonstrates the use of a **LoRA** (Low-Rank Adaptation) technique to fine-tune a pretrained model (`Llama-2-7b-chat-hf`) for specific tasks. We compare the output of the **base model** and **LoRA model** when asked a legal-related question from the case of **Smith v. Greenfield Corporation**.

## Setup

To run this project, make sure you have the following dependencies installed:

* `transformers`
* `torch`
* `peft`
* `bitsandbytes`

You can install them using pip:

```bash
pip install transformers torch peft bitsandbytes
```

### Loading the Models

1. **Base Model**: The `meta-llama/Llama-2-7b-chat-hf` is a pretrained model from Hugging Face.
2. **LoRA Model**: This model is the `Llama-2-7b-chat-hf` fine-tuned with **LoRA** layers, which adapt the model with specific knowledge learned through additional training.

The models are loaded using the following code:

```python
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
```

### Test Prompt

We test the model by providing a case-based legal question:

**Prompt**:
"In the case of Smith v. Greenfield Corporation, why did the court rule in favor of Jane Smith despite Greenfield Corporation claiming the software was defective?"

### Output Comparison

#### Base Model Output:

```text
In the case of Smith v. Greenfield Corporation, why did the court rule in favor of Jane Smith despite Greenfield Corporation claiming the software was defective?
In the case of Smith v. Greenfield Corporation, the court ruled in favor of Jane Smith because the court found that Greenfield Corporation had a duty to warn Smith of the potential dangers of the software. The court held that the software was not defective in and of itself, but that Greenfield Corporation had a duty to take reasonable steps to ensure that Smith was aware of the potential risks associated with the software.

In reaching this decision, the court considered several factors, including
```

#### LoRA Model Output:

```text
In the case of Smith v. Greenfield Corporation, why did the court rule in favor of Jane Smith despite Greenfield Corporation claiming the software was defective?
Jane Smith, a software developer, entered into a contract with Greenfield Corporation to develop a custom inventory management system. The contract stipulated a payment of $50,000 upon completion of the project. However, upon delivery, Greenfield Corporation claimed the software was defective and refused to pay.
Smith argued that the software met all specified requirements and was functionally sound. The court, after reviewing the contract and evaluating the delivered software, ruled in favor
```

### Key Differences:

* **Base Model Output**:
  The base model introduces concepts about **duty of warning** and **risks** related to the software, which are not explicitly mentioned in the case summary. It also includes incomplete text that might be due to random text generation.

* **LoRA Model Output**:
  The LoRA-enhanced model provides a more **contextually relevant response**, referencing the **contract** and **payment stipulations** between Jane Smith and Greenfield Corporation. The output cuts off, indicating it might be generated in a real-time application but shows a much more focused response compared to the base model.

---

## Conclusion

This comparison demonstrates how **LoRA** enhances the base model by applying specialized knowledge from the training process, providing more accurate and context-driven responses. The LoRA model's response is more coherent with the provided legal case context, reflecting the model's improved performance due to fine-tuning.

---

Let me know if you need any additional details or refinements!
