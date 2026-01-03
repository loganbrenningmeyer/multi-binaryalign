from transformers import AutoTokenizer, AutoModel

model_name = "microsoft/mdeberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)