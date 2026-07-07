"""
STEP 3: Search the image index using a TEXT query.
Must run 2_build_search_index.py first.

Usage:
    python 3_search_images.py "a dog playing on the beach"
"""

import sys
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel

INDEX_FILE = "image_index.faiss"
PATHS_FILE = "image_paths.npy"
TOP_K = 5

def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def search(query_text, model, processor):
    index = faiss.read_index(INDEX_FILE)
    image_paths = np.load(PATHS_FILE)

    inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    text_features = model.get_text_features(**inputs).pooler_output
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    query_vector = text_features.detach().numpy().astype("float32")

    scores, indices = index.search(query_vector, TOP_K)

    print(f"\nTop {TOP_K} results for query: '{query_text}'\n")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        print(f"{rank}. {image_paths[idx]}  (similarity: {score:.4f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python 3_search_images.py "your text query"')
        sys.exit(1)

    model, processor = load_clip()
    search(sys.argv[1], model, processor)
