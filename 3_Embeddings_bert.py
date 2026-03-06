from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "The bank will not approve the loan because the credit score is low"

inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)

print(outputs.last_hidden_state.shape)

#Trying with 2 Example
text1 = "He sat by the bank of the river."
text2 = "He deposited money in the bank."

inputs1 = tokenizer(text1, return_tensors="pt")
inputs2 = tokenizer(text2, return_tensors="pt")

out1 = model(**inputs1)
out2 = model(**inputs2)

bank_vec1 = out1.last_hidden_state[0][4]  # position of 'bank'
bank_vec2 = out2.last_hidden_state[0][5]

print(torch.cosine_similarity(bank_vec1, bank_vec2, dim=0))
