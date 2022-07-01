# Bi-encoder entity linking model

Simple, but efficient model to retrieve entities in the given piece of text. The model uses BERT encoder for text and entities and fast HNSW index to retrieve nearest entities.

## Usage:

```
python main.py "Nobel Prize-winning physicist who developed the theory of general relativity."
```

## Train the model:

### Download wikipedia dump

First we need to download and wikipedia data:
```
scripts/download_wiki.sh
```

```
python scripts/preprocess_wiki.py
```

### Train the model

```
python train.py
```

### Generate entities embeddings index

```
python generate_entity_embeddings.py
```