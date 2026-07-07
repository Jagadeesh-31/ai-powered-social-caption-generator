"""
STEP 2: Build a searchable image index using CLIP + FAISS.
Point this at a folder of images. It creates embeddings for every image
and saves a FAISS index so you can search images using a TEXT query later.

Usage:
    python 2_build_search_index.py ./data/images
"""

import sys
import os
import glob
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

INDEX_FILE = "image_index.faiss"
PATHS_FILE = "image_paths.npy"

def load_clip():
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def build_index(image_folder, model, processor):
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(image_folder, ext)))

    if not image_paths:
        print(f"No images found in {image_folder}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images. Generating embeddings...")

    embeddings = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        image_features = model.get_image_features(**inputs).pooler_output
        # normalize so we can use cosine similarity via inner product
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embeddings.append(image_features.detach().numpy()[0])

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine sim (since normalized)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    np.save(PATHS_FILE, np.array(image_paths))

    print(f"Index built and saved: {INDEX_FILE}, {PATHS_FILE}")
    print(f"Total images indexed: {len(image_paths)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 2_build_search_index.py <image_folder>")
        sys.exit(1)

    model, processor = load_clip()
    build_index(sys.argv[1], model, processor)
