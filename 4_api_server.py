"""
STEP 4: Wrap captioning in a simple API so a frontend or client app can call it.

Run:
    uvicorn 4_api_server:app --reload --port 8000

Then test at:
    http://127.0.0.1:8000/docs   (interactive Swagger UI)
"""

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration

app = FastAPI(title="Image Captioning API")

print("Loading model, please wait...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Model loaded. Server ready.")

@app.get("/")
def root():
    return {"status": "running", "endpoint": "/generate-caption"}

@app.post("/generate-caption")
async def generate_caption(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return {"filename": file.filename, "caption": caption}
