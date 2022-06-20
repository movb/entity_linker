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

# Initialize the tokenizer and entity encoder
entity_model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(entity_model_name)
entity_encoder = AutoModel.from_pretrained('bi_encoder_output/entity_encoder')

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
index = faiss.IndexHNSWFlat(embedding_size, 32)
index.hnsw.efConstruction = 40
index.verbose = True

# Add the embeddings to the Faiss index
faiss.normalize_L2(entity_embeddings.numpy())
index.add(entity_embeddings.numpy())

# Save the Faiss index
faiss.write_index(index, 'entity_embeddings_hnsw.faiss')