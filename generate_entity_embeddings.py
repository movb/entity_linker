import json
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

from .config import config

# Load the JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

file_path = config['data']['file_path']
data = load_json_data(file_path)

# Initialize the tokenizer and entity encoder
entity_model_name = config['model']['entity_model_name']
tokenizer = AutoTokenizer.from_pretrained(entity_model_name)
entity_encoder = AutoModel.from_pretrained(config['model']['entity_encoder_path'])

def generate_entity_embeddings(data, tokenizer, entity_encoder):
    entity_embeddings = []

    for page in data:
        entity_text = page['text']
        encoded_input = tokenizer(entity_text, truncation=True, padding=True, return_tensors="pt")

        with torch.no_grad():
            embeddings = entity_encoder(**encoded_input).last_hidden_state[:, 0, :]

        entity_embeddings.append(embeddings)

    return torch.cat(entity_embeddings, dim=0)

entity_embeddings = generate_entity_embeddings(data, tokenizer, entity_encoder)

# Save the generated embeddings to a torch file
torch.save(entity_embeddings, 'entity_embeddings.pt')

# Initialize the Faiss index
embedding_size = entity_embeddings.shape[1]
index = faiss.IndexHNSWFlat(embedding_size, 32)
index.hnsw.efConstruction = 40
index.verbose = True

# Add the embeddings to the Faiss index
faiss.normalize_L2(entity_embeddings.numpy())
index.add(entity_embeddings.numpy())

# Save the Faiss index
faiss.write_index(index, 'entity_embeddings_hnsw.faiss')