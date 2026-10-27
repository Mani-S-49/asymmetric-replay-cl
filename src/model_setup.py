import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def get_quantized_model_and_tokenizer(model_id: str = "meta-llama/Llama-3.2-1B"):
    print(f"Loading base model: {model_id}")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    print("Model successfully loaded with QLoRA configuration!")
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = get_quantized_model_and_tokenizer()
