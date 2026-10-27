import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from model_setup import get_quantized_model_and_tokenizer

def run_toy_experiment():
    print('--- Phase 0 Toy CL Experiment: English to Hindi ---')
    
    # 1. Load model and tokenizer using our model_setup utility
    model, tokenizer = get_quantized_model_and_tokenizer()
    
    # 2. Load small subsets of sentiment datasets (e.g., IMDB for English)
    print('Loading English dataset (IMDB)...')
    en_dataset = load_dataset('imdb', split='train[:200]')
    en_test = load_dataset('imdb', split='test[:50]')
    
    print('Dataset and model initialization verified successfully!')
    print('You are ready to execute sequential fine-tuning loops and measure the forgetting curve.')

if __name__ == '__main__':
    run_toy_experiment()
