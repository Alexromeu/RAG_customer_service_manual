import numpy as np
import ollama
import json



def get_embedding(text: str) -> np.ndarray:
    response = ollama.embed(model='embeddinggemma', input=text)
    return np.array(response['embeddings'][0])

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def embedd_documents(json_url: str) -> np.ndarray:
    resulting_matrix = []
    with open(json_url, "r") as file:
        data = json.load(file)

    for chunk in data:
        resulting_matrix.append(get_embedding(chunk["text"]))

    return np.array(resulting_matrix)








def main():
    json_data = embedd_documents("./data/chunks.json")
    for chunk in json_data:
        print(chunk, '\n')

if __name__ == "__main__":
    main()
    print("success")




'''
example of json data
{
  "id": 3,
  "chapter": 3,
  "section": "3.6.3",
  "title": "Notification of FAM Transport to AA Station Personnel 3-42",
  "text": "Section 3.6.3 — Notification of FAM Transport to AA Station Personnel 3-42\n\n3.6.4 3.6.5 3.6.6 3.6.7 3.7 3.7.1 3.7.2 3.8 3.8.1 3.8.2 3.8.3 3.8.4 3.8.5 3.9"
 },'''