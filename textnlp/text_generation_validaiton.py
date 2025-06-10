
# Synthetic Text Generation & Validation with BLEU and ROUGE

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_metric
import torch

# Step 1: Load Pretrained Model & Tokenizer
model_name = "gpt2"  # Replace with EleutherAI/gpt-j-6B or gpt-neox if available
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Step 2: Define Prompt and Generate Text
prompt = "In the future, synthetic data will revolutionize"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, do_sample=True, top_k=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 3: Define Reference Text for Evaluation
reference = "In the future, synthetic data will revolutionize machine learning by enabling safer and more diverse training."

# Step 4: BLEU Score Calculation
bleu = load_metric("bleu")
generated_tokens = generated_text.split()
reference_tokens = [reference.split()]
bleu_result = bleu.compute(predictions=[generated_tokens], references=[reference_tokens])

# Step 5: ROUGE Score Calculation
rouge = load_metric("rouge")
rouge_result = rouge.compute(predictions=[generated_text], references=[reference])

# Step 6: Display Results
print("Generated Text:\n", generated_text)
print("\nBLEU Score:\n", bleu_result)
print("\nROUGE Score:\n", rouge_result)

# pip install transformers datasets torch


