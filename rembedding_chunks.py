import requests
import json
import pandas as pd
import os
import numpy as np
import joblib
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed",json={
        "model": "bge-m3",#BAAI General Embedding Beijing Academy of Artificial Intelligence.....XLM-RoBERTa 560M parameters
        "input": text_list
    })
    
    embedding = r.json()["embeddings"]
    return embedding
df = joblib.load("embedding.joblib")
print(df)
# a = create_embedding("Hello, how are you?")
# print(a)

# dataset = os.listdir("jsons")
# chunk_id = 0
# my_dic = []
# for data in dataset:
#     with open(f"jsons/{data}") as f:
#         chunks = json.load(f)

#         embedding = create_embedding([chunk["text"] for chunk in chunks[0]["chunks"]])
#     for i,chunk in enumerate(chunks[0]["chunks"]):
#         chunk["chunk_id"] = chunk_id 
#         chunk["embedding"] = embedding[i]
#         chunk_id += 1
#         my_dic.append(chunk) 


# df = pd.DataFrame.from_records(my_dic)
# joblib.dump(df,"embedding.joblib")

# incomming_querry = input("write your querry: ")
# querry_embedding = create_embedding(incomming_querry)[0]
# #print(df["embedding"].values)
# # print(np.vstack(df["embedding"].values).shape) 
# similarties = cosine_similarity(np.vstack(df["embedding"].values),[querry_embedding]).flatten()
# print(similarties)
# similarties = np.argsort(similarties)[::-1][0:3]
# print(similarties)