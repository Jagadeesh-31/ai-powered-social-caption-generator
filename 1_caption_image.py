"""
STEP 1: Basic image captioning using BLIP.
Run this first to confirm the model works on your machine.

Usage:
    python 1_caption_image.py path/to/image.jpg
"""

import sys
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_model():
    print("Loading BLIP model (first run will download ~1GB, be patient)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_caption(image_path, processor, model):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 1_caption_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    processor, model = load_model()
    caption = generate_caption(image_path, processor, model)
    print(f"\nImage: {image_path}")
    print(f"Caption: {caption}")
