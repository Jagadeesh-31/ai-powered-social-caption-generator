import os
import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from google import genai
from google.genai.errors import ServerError, ClientError

app = FastAPI(title="Gemini Social Caption API")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        "style": "conversational, storytelling tone; minimal hashtags (0-2); can ask a question to invite comments",
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

def generate_content_with_retry(client, model, contents, max_retries=4, delay=2):
    import time
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

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def read_index():
    try:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
        if not os.path.exists(path):
            path = "index.html"
        with open(path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return HTMLResponse(content=f"Error loading index.html: {str(e)}", status_code=500)

@app.get("/api/health")
def health():
    return {"status": "healthy", "service": "Gemini Social Caption API"}

@app.post("/api/generate-captions")
async def generate_captions(
    file: UploadFile = File(...),
    platforms: str = Form(...),
    tone: str = Form("Friendly and upbeat"),
    context: str = Form(""),
    api_key: str = Form("")
):
    # Determine API Key: use user-supplied or fallback to environment variable
    key = api_key.strip() or os.environ.get("GEMINI_API_KEY")
    if not key:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Gemini API Key is required. Please set the GEMINI_API_KEY environment variable or provide it in the input field.")
    client = genai.Client(api_key=key)

    # Load PIL image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 1. Analyze the image to get a factual description using Gemini Vision
    image_description = generate_content_with_retry(
        client=client,
        model="gemini-3.5-flash",
        contents=[image, "Provide a plain, factual, one-sentence description of this image."]
    )

    # 2. Iterate through requested platforms and write captions
    platform_list = [p.strip().lower() for p in platforms.split(",")]
    results = {}

    for platform_key in platform_list:
        if platform_key not in PLATFORM_RULES:
            continue
        rules = PLATFORM_RULES[platform_key]
        context_line = f"\nAdditional context from the user: {context}\n" if context else ""
        
        prompt = f"""You are a social media copywriter. Here is a plain, factual description
of an image: "{image_description}"
{context_line}
Write ONE ready-to-post caption for {rules['label']}.

Requirements:
- Tone: {tone}
- Platform style: {rules['style']}
- Hard limit: must be under {rules['max_chars']} characters, including hashtags
- Output ONLY the caption text. No explanation, no quotation marks, no preamble.
"""
        caption = generate_content_with_retry(
            client=client,
            model="gemini-3.5-flash",
            contents=prompt
        )
        results[platform_key] = {
            "label": rules["label"],
            "caption": caption,
            "max_chars": rules["max_chars"]
        }

    return {
        "description": image_description,
        "captions": results
    }
