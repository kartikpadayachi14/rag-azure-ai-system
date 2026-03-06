from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Artificial Intelligence is transforming the world."

tokens = tokenizer(text)

print(tokens)

decoded = tokenizer.decode(tokens["input_ids"])

print(decoded)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

inputs = tokenizer(text, return_tensors="pt")

output = model.generate(inputs["input_ids"], max_length=100)

summary = tokenizer.decode(output[0], skip_special_tokens=True)

print(summary)
