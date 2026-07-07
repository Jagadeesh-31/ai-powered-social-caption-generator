import os
import io
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    temp_file_path = None
    uploaded_file = None
    client = None
    try:
        # Determine API Key: use user-supplied or fallback to environment variable
        key = api_key.strip() or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise HTTPException(
                status_code=400,
                detail="Gemini API Key is required. Please set the GEMINI_API_KEY environment variable or provide it in the settings."
            )
        client = genai.Client(api_key=key)

        is_video = file.content_type and file.content_type.startswith("video/")
        media_type = "video" if is_video else "image"

        if is_video:
            # Create a temporary file to save the video upload
            file_extension = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name

            # Upload video to Gemini using File API
            uploaded_file = client.files.upload(file=temp_file_path)

            # Poll until video is ACTIVE
            import time
            max_polls = 45
            poll_count = 0
            while uploaded_file.state.name == "PROCESSING" and poll_count < max_polls:
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)
                poll_count += 1

            if uploaded_file.state.name == "FAILED":
                raise Exception(f"Video processing failed on Gemini: {uploaded_file.error.message if hasattr(uploaded_file, 'error') else 'Unknown error'}")
            if uploaded_file.state.name != "ACTIVE":
                raise Exception("Video processing timed out on Gemini. Please try again.")

            # Get video description
            image_description = generate_content_with_retry(
                client=client,
                model="gemini-3.5-flash",
                contents=[uploaded_file, "Provide a plain, factual, one-sentence description of this video."]
            )
        else:
            # Process as Image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

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
of a {media_type}: "{image_description}"
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

    except Exception as e:
        # Log error trace for debugging
        import traceback
        traceback.print_exc()

        if isinstance(e, HTTPException):
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        
        err_msg = str(e)
        status_code = 400
        # If API key is wrong, genai SDK usually raises ClientError containing API_KEY_INVALID or 403
        if "api_key" in err_msg.lower() or "api key" in err_msg.lower() or "403" in err_msg or "unauthorized" in err_msg.lower():
            err_msg = "Invalid Gemini API Key. Please verify your API Key in the settings."
            status_code = 400
        elif "quota" in err_msg.lower() or "rate limit" in err_msg.lower() or "429" in err_msg:
            err_msg = "Gemini API rate limit exceeded. Please try again shortly."
            status_code = 429

        return JSONResponse(status_code=status_code, content={"detail": err_msg})

    finally:
        # Cleanup temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
        # Cleanup uploaded file from Gemini File API
        if uploaded_file and client:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

