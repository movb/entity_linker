import json
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

# Load the JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

file_path = 'wikipedia/wikipedia_data.json'
data = load_json_data(file_path)

# Load the trained model, tokenizer, and Faiss index
text_model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_encoder = AutoModel.from_pretrained('bi_encoder_output/text_encoder')

index = faiss.read_index('entity_embeddings_hnsw.faiss')

# Define a function to preprocess and embed the input text
def embed_text(text, tokenizer, text_encoder):
    encoded_input = tokenizer(text, truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        embeddings = text_encoder(**encoded_input).last_hidden_state[:, 0, :]

    return embeddings

# Run the inference and search for the nearest entities
def search_nearest_entities(text, tokenizer, text_encoder, index, data, k=5):
    text_embeddings = embed_text(text, tokenizer, text_encoder)
    faiss.normalize_L2(text_embeddings.numpy())
    _, indices = index.search(text_embeddings.numpy(), k)

    nearest_entities = [data[i]['entity'] for i in indices[0]]

    return nearest_entities

input_text = "Nobel Prize-winning physicist who developed the theory of general relativity."
nearest_entities = search_nearest_entities(input_text, tokenizer, text_encoder, index, data)

# Print the nearest entity names
print("Nearest Entities:")
for entity in nearest_entities:
    print(f"- {entity}")