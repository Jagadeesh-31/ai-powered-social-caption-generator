import os
import io
import tempfile
import shutil
import random
from typing import Optional
from pydantic import BaseModel
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
    "youtube": {
        "label": "YouTube",
        "max_chars": 5000,
        "style": "video-description style, engaging layout with paragraphs and bullet points, includes standard call-to-action like 'subscribe and turn on notifications', 3-5 tags at the bottom",
    },
    "kick": {
        "label": "Kick",
        "max_chars": 150,
        "style": "gamer stream title, high energy, attention-grabbing hook to join stream, 2-3 emojis, gamer slang friendly, 1-2 hashtags",
    },
}

class SocialCaptions(BaseModel):
    instagram: Optional[str] = None
    x: Optional[str] = None
    facebook: Optional[str] = None
    linkedin: Optional[str] = None
    pinterest: Optional[str] = None
    youtube: Optional[str] = None
    kick: Optional[str] = None

def generate_content_with_retry(client, model, contents, max_retries=5, **kwargs):
    import time
    
    # Define fallback chain for models
    model_fallbacks = [model]
    if model == "gemini-3.5-flash":
        model_fallbacks.extend(["gemini-2.5-flash", "gemini-2.0-flash", "gemini-3.1-flash-lite"])
        
    last_exception = None
    for current_model in model_fallbacks:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    **kwargs
                )
                return response
            except Exception as e:
                last_exception = e
                err_str = str(e).upper()
                is_busy = "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "LIMIT" in err_str or "BUSY" in err_str
                
                if is_busy:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"Model {current_model} busy, retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"Model {current_model} exhausted retries. Trying fallback model if available...")
                        break  # Try the next fallback model in the list
                else:
                    # If the model is not found or not permitted, and we have other fallbacks left, try the next fallback model.
                    is_model_unavailable = "NOT_FOUND" in err_str or "404" in err_str or "PERMISSION" in err_str or "403" in err_str or "INVALID_ARGUMENT" in err_str
                    if is_model_unavailable and current_model != model_fallbacks[-1]:
                        print(f"Model {current_model} not accessible, trying fallback model...")
                        break
                    raise e  # Non-retryable error (e.g. invalid API key), raise immediately
                    
    if last_exception:
        raise last_exception
    raise Exception("Gemini API still unavailable after trying fallbacks.")


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
                detail="Gemini API Key is required. Please set the GEMINI_API_KEY environment variable or provide it in the developer settings."
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
            description_resp = generate_content_with_retry(
                client=client,
                model="gemini-3.5-flash",
                contents=[uploaded_file, "Provide a plain, factual, one-sentence description of this video."]
            )
            image_description = description_resp.text.strip()
        else:
            # Process as Image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            description_resp = generate_content_with_retry(
                client=client,
                model="gemini-3.5-flash",
                contents=[image, "Provide a plain, factual, one-sentence description of this image."]
            )
            image_description = description_resp.text.strip()

        # 2. Generate captions in a single prompt using structured JSON output
        platform_list = [p.strip().lower() for p in platforms.split(",")]
        results = {}

        rules_text_list = []
        for platform_key in platform_list:
            if platform_key in PLATFORM_RULES:
                rules = PLATFORM_RULES[platform_key]
                rules_text_list.append(f"- {rules['label']}: Style: {rules['style']}, Max characters: {rules['max_chars']}")
        rules_text = "\n".join(rules_text_list)
        
        context_line = f"\nAdditional context from the user: {context}\n" if context else ""

        prompt = f"""You are a social media copywriter. Here is a plain, factual description
of a {media_type}: "{image_description}"
{context_line}

Write ready-to-post captions for the following platforms, matching their specific styles and character limits:
{rules_text}

Requirements:
- Tone: The tone MUST be "{tone}" for all generated captions. Ensure this tone is strongly reflected in the writing style.
- Completeness: You MUST generate a custom caption for every single platform requested above. Do not leave any requested platform blank or null.
- Hard limit: Each caption must be strictly under the character limit specified for its platform (including hashtags).
- Output: Output MUST be a JSON object containing the caption for each platform.
"""
        from google.genai import types
        
        response = generate_content_with_retry(
            client=client,
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SocialCaptions,
            )
        )

        # Parse structured output safely
        caption_text_map = {}
        try:
            if response.parsed:
                if hasattr(response.parsed, "model_dump"):
                    caption_text_map = response.parsed.model_dump()
                else:
                    caption_text_map = response.parsed.__dict__
            else:
                import json
                caption_text_map = json.loads(response.text)
        except Exception:
            try:
                import json
                caption_text_map = json.loads(response.text)
            except Exception:
                caption_text_map = {}

        for platform_key in platform_list:
            if platform_key not in PLATFORM_RULES:
                continue
            rules = PLATFORM_RULES[platform_key]
            caption_text = caption_text_map.get(platform_key) or ""
            
            # If returned as non-string, cast to string
            if caption_text is None:
                caption_text = ""
            elif not isinstance(caption_text, str):
                caption_text = str(caption_text)
                
            results[platform_key] = {
                "label": rules["label"],
                "caption": caption_text.strip(),
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
            err_msg = "Invalid Gemini API Key. Please verify your API Key in the developer settings."
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

