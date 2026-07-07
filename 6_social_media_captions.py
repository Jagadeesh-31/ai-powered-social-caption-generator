"""
Generate ready-to-post social media captions from an image, for
multiple platforms at once (Instagram, X/Twitter, Facebook, LinkedIn, Pinterest) using Gemini.

Before running, set your Gemini API key as an environment variable:
    Windows (PowerShell):  $env:GEMINI_API_KEY="your-key-here"
    Mac/Linux:             export GEMINI_API_KEY="your-key-here"

Usage:
    python 6_social_media_captions.py path/to/image.jpg
    python 6_social_media_captions.py path/to/image.jpg --tone playful --platforms instagram,x,linkedin
"""

import sys
import argparse
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from google import genai

PLATFORM_RULES = {
    "instagram": {
        "label": "Instagram",
        "max_chars": 2200,
        "style": "warm, visual, engaging; 3-5 relevant hashtags at the end; 1-2 emojis max; can include a short call-to-action like 'double tap' or 'tag a friend'",
    },
    "x": {
        "label": "X (Twitter)",
        "max_chars": 280,
        "style": "punchy, witty, very concise; 1-2 hashtags max; must fit in 280 characters INCLUDING hashtags",
    },
    "facebook": {
        "label": "Facebook",
        "max_chars": 500,
        "style": "conversational, slightly longer-form, storytelling tone; minimal hashtags (0-2); can ask a question to invite comments",
    },
    "linkedin": {
        "label": "LinkedIn",
        "max_chars": 700,
        "style": "professional but human, insight or takeaway oriented, no cutesy emojis; 2-3 professional hashtags relevant to industry/career",
    },
    "pinterest": {
        "label": "Pinterest",
        "max_chars": 500,
        "style": "descriptive and keyword-rich for searchability, inspirational tone; include practical/how-to angle if relevant; 3-5 hashtags",
    },
}

def load_blip():
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def get_image_description(image_path, processor, model):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)

def build_prompt(image_description, platform_key, tone, extra_context):
    rules = PLATFORM_RULES[platform_key]
    context_line = f"\nAdditional context from the user about this post: {extra_context}\n" if extra_context else ""

    prompt = f"""You are a social media copywriter. Here is a plain, factual description
of an image (from an image captioning model): "{image_description}"
{context_line}
Write ONE ready-to-post caption for {rules['label']}.

Requirements:
- Tone: {tone}
- Platform style: {rules['style']}
- Hard limit: must be under {rules['max_chars']} characters, including hashtags
- Output ONLY the caption text (and hashtags if applicable). No explanation, no quotation marks, no preamble.
"""
    return prompt

import time
from google.genai.errors import ServerError, ClientError

def generate_content_with_retry(client, model, contents, max_retries=4, delay=2):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents
            )
            return response.text.strip()
        except (ServerError, ClientError) as e:
            is_retryable = isinstance(e, ServerError) or (isinstance(e, ClientError) and getattr(e, "code", None) == 429)
            if is_retryable and attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                raise e

def generate_caption_for_platform(client, image_description, platform_key, tone, extra_context):
    prompt = build_prompt(image_description, platform_key, tone, extra_context)
    return generate_content_with_retry(
        client=client,
        model="gemini-3.5-flash",
        contents=prompt
    )

def main():
    parser = argparse.ArgumentParser(description="Generate social media captions from an image using Gemini.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--platforms", default="instagram,x,facebook,linkedin,pinterest",
                         help="Comma-separated list of platforms (default: all)")
    parser.add_argument("--tone", default="friendly and upbeat",
                         help="Overall tone, e.g. 'playful', 'professional', 'inspirational', 'humorous'")
    parser.add_argument("--context", default="",
                         help="Optional extra context, e.g. 'this is for our new product launch'")
    args = parser.parse_args()

    platforms = [p.strip().lower() for p in args.platforms.split(",")]
    for p in platforms:
        if p not in PLATFORM_RULES:
            print(f"Unknown platform '{p}'. Valid options: {', '.join(PLATFORM_RULES.keys())}")
            sys.exit(1)

    processor, model = load_blip()
    image_description = get_image_description(args.image_path, processor, model)
    print(f"\nImage description (from BLIP): {image_description}\n")

    import os
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not found.")
        print("Please set it before running this script. E.g.:")
        print("  Windows: $env:GEMINI_API_KEY=\"your_key_here\"")
        print("  Mac/Linux: export GEMINI_API_KEY=\"your_key_here\"")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print("=" * 60)
    for platform_key in platforms:
        label = PLATFORM_RULES[platform_key]["label"]
        caption = generate_caption_for_platform(client, image_description, platform_key, args.tone, args.context)
        print(f"\n📱 {label} caption:\n{caption}\n")
        print(f"(length: {len(caption)} characters, limit: {PLATFORM_RULES[platform_key]['max_chars']})")
        print("-" * 60)

if __name__ == "__main__":
    main()
