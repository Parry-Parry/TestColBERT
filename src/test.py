from model.colbert import ColBERT
import transformers
import torch
from transformers import AutoTokenizer
from util.train_util import init_tokenized

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # honestly not sure if that is the best way to go, but it works :)
model = ColBERT.from_pretrained("sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco")

collate_func = init_tokenized(tokenizer)

ds = 

