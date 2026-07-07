"""
Client-facing demo UI — upload an image, pick platforms and tone,
get ready-to-post captions for all of them at once using Gemini.

Before running, set your Gemini API key as an environment variable:
    Windows (PowerShell):  $env:GEMINI_API_KEY="your-key-here"
    Mac/Linux:             export GEMINI_API_KEY="your-key-here"

Run:
    streamlit run 7_streamlit_social_demo.py
"""

import os
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from google import genai

PLATFORM_RULES = {
    "Instagram": {"key": "instagram", "max_chars": 2200,
        "style": "warm, visual, engaging; 3-5 relevant hashtags at the end; 1-2 emojis max"},
    "X (Twitter)": {"key": "x", "max_chars": 280,
        "style": "punchy, witty, very concise; 1-2 hashtags max; must fit in 280 characters total"},
    "Facebook": {"key": "facebook", "max_chars": 500,
        "style": "conversational, storytelling tone; minimal hashtags (0-2); can invite comments"},
    "LinkedIn": {"key": "linkedin", "max_chars": 700,
        "style": "professional but human, insight-oriented, no cutesy emojis; 2-3 professional hashtags"},
    "Pinterest": {"key": "pinterest", "max_chars": 500,
        "style": "descriptive, keyword-rich, inspirational; 3-5 hashtags"},
}

st.set_page_config(page_title="Social Media Caption Generator", layout="centered")
st.title("📱 Social Media Caption Generator")
st.write("Upload an image → get ready-to-post captions for every platform.")

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip()

# API Key Validation and Client Initialization
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.sidebar.warning("⚠️ GEMINI_API_KEY environment variable not found.")
    api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

if not api_key:
    st.info("👉 Please enter your Gemini API Key in the sidebar to generate captions.")
    st.stop()

client = genai.Client(api_key=api_key)

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
            # Retry on 503 (Server Error) or 429 (Rate Limit / Client Error with code 429)
            is_retryable = isinstance(e, ServerError) or (isinstance(e, ClientError) and getattr(e, "code", None) == 429)
            if is_retryable and attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
            else:
                raise e


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    selected_platforms = st.multiselect(
        "Platforms", list(PLATFORM_RULES.keys()), default=list(PLATFORM_RULES.keys())
    )
with col2:
    tone = st.selectbox(
        "Tone", ["Friendly and upbeat", "Playful", "Professional", "Inspirational", "Humorous", "Minimal/aesthetic"]
    )

extra_context = st.text_input("Optional context (e.g. 'new product launch', 'weekend getaway')", "")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Captions", type="primary"):
        with st.spinner("Analyzing image..."):
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=30)
            image_description = processor.decode(out[0], skip_special_tokens=True)

        st.caption(f"Image understood as: *{image_description}*")

        for platform_name in selected_platforms:
            rules = PLATFORM_RULES[platform_name]
            context_line = f"\nAdditional context: {extra_context}\n" if extra_context else ""

            prompt = f"""You are a social media copywriter. Here is a plain, factual description
of an image: "{image_description}"
{context_line}
Write ONE ready-to-post caption for {platform_name}.

Requirements:
- Tone: {tone}
- Platform style: {rules['style']}
- Hard limit: must be under {rules['max_chars']} characters, including hashtags
- Output ONLY the caption text. No explanation, no quotation marks, no preamble.
"""
            with st.spinner(f"Writing {platform_name} caption..."):
                caption = generate_content_with_retry(
                    client=client,
                    model="gemini-3.5-flash",
                    contents=prompt
                )

            st.subheader(platform_name)
            st.text_area(f"{platform_name} caption", caption, height=120, label_visibility="collapsed")
            st.caption(f"{len(caption)} / {rules['max_chars']} characters")
            st.divider()
