import numpy as np
import ollama
import json
from typing import List

class Embedding:
    def __init__(self):
        self.database: List[np.ndarray] = []
        self.raw_data = []
        

    def get_embedding(self, text: str) -> np.ndarray:
        response = ollama.embed(model='embeddinggemma', input=text)
        return np.array(response['embeddings'][0])

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


    def embedd_documents(self, json_path: str):
        resulting_matrix = []
        with open(json_path, "r") as file:
            data = json.load(file)
            self.raw_data = data

        for chunk in data:
            resulting_matrix.append(self.get_embedding(chunk["text"]))

        self.database = resulting_matrix



    def extract_similar_data(self, input: str) -> str:
        if not len(self.database): 
            return "FILL OUT THE DATABASE FIRST with self.embedd_documents()"

        similaritys= []

        input_embedding = self.get_embedding(input)
        for vector in self.database:
            similaritys.append(self.cosine_similarity(vector, input_embedding))

        similaritys = np.array(similaritys)

        # 1. Use argsort to get indices from lowest to highest score
        sorted_indices = np.argsort(similaritys) 
        
        # 2. Slice the last '5' elements and reverse them (highest scores first)
        top_indices = sorted_indices[-5:][::-1]

        # 3. Combine the top matching text chunks into one context block
        retrieved_contexts = []
        for rank, index in enumerate(top_indices, 1):
            chunk_text = self.raw_data[index]["text"]
            score = similaritys[index]
            retrieved_contexts.append(f"[Match {rank} | Score: {score:.4f}]\n{chunk_text}")

        return "\n\n---\n\n".join(retrieved_contexts)


         



def main():
    emb = Embedding()
    emb.embedd_documents("./data/chunks.json")
    print(emb.extract_similar_data("what is a dengerous good?"))
   

if __name__ == "__main__":
    main()
    print("success")


'''
    first embed the data in json with self.embedd_documents(url)
    then extract the simmilar data, so from an imput you get whatever in the database looks similar
'''

'''
example of json data
{
  "id": 3,
  "chapter": 3,
  "section": "3.6.3",
  "title": "Notification of FAM Transport to AA Station Personnel 3-42",
  "text": "Section 3.6.3 — Notification of FAM Transport to AA Station Personnel 3-42\n\n3.6.4 3.6.5 3.6.6 3.6.7 3.7 3.7.1 3.7.2 3.8 3.8.1 3.8.2 3.8.3 3.8.4 3.8.5 3.9"
 },'''