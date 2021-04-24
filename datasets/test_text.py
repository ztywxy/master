from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertTokenizer
import os
import pickle
from tqdm import tqdm
import torch
from text_data_utils import chunks
import pandas as pd

def read_in_chunks(filePath, chunk_size=1024*8):
    file_object = open(filePath, encoding="utf-8")
    while True:
        chunk_data = file_object.read(chunk_size)
        if not chunk_data:
            break
        yield chunk_data

filePath = '/home/teddy/nlp/knowledge-probing-private-tianyi/data/training_data/PAQ/paq_train.tsv'
fsize = os.path.getsize(filePath)
print(fsize)
#for chunk in read_in_chunks(filePath):
#    text_chunks = list(chunks(chunk, 5))
#    print(text_chunks)


#with open('/home/teddy/nlp/knowledge-probing-private-tianyi/data/training_data/PAQ/test.tsv', encoding="utf-8") as f:
#    text = f.read()
#text_chunks = list(chunks(text, 5))
# #print('!!!!!!!!!!'*20)
#print(text_chunks)
# examples = []
# for chunk in tqdm(text_chunks):
#     #print(chunk)
#     print('------------------')
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     batch = tokenizer(
#         chunk, truncation=True, padding='max_length', return_overflowing_tokens=True)
#     #print(batch)
#     for ids in batch['input_ids']:
#         examples.append(ids)
# print(examples)
