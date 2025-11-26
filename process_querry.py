import joblib
import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed",json={
        "model": "bge-m3",
        "input": text_list
    })
    
    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate",json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    
    response = r.json()
    #print(response)
    return response
    

df = joblib.load("embedding.joblib")

querry = input("Ask querry: ")
querry_embedding = create_embedding(querry)[0]

similarities = cosine_similarity(np.vstack(df["embedding"].values), [querry_embedding]).flatten()
top_similarity = 5
top_indx = np.argsort(similarities)[::-1][:top_similarity]

new_df = df.loc[top_indx]

prompt = f'''I am teaching web development by using sigma web development course. Here are the video subtitle chunks which containing video title, video number, start time in seconds, end time in seconds, the text at that time.

{new_df[["title","number","start","end","text"]].to_json(orient = "records")}

---------------------------------------------
'{querry}' This is the querry from the user. You have to answer where and how much the content taught in which video from start time to end time (and at what time frame) and guid the user to that video chunk. If user asked unrelated question then respond the user that i can only answer the question related  to the course. Don't show your thought process, only give the final answer.
'''  
with open("promt.txt", "w") as f:
    f.write(prompt)
    
# for index, item in new_df.iterrows():
#     print(index,item["title"],item["number"],item["text"],item["start"],item["end"])
response = inference(prompt)["response"]
print(inference(prompt)["response"])

with open("response.text","w") as f:
    data = f.write(response)
