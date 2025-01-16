from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import base64
from openai import OpenAI
import os

app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key="sk-None-oezPmoiVxf7FPSqYyahPT3BlbkFJ0Gtet94cqqCiN5SFAmKf")

# Function to encode the image as base64
def encode_image(file: UploadFile):
    try:
        # Read the file's content
        content = file.file.read()
        # Convert to base64
        return base64.b64encode(content).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")
    

SYSTEM_PROMPT = """
Analyze the given image of a backyard. Provide detailed insights on the following:

1. Identify and describe all the visible flowers, including their types, colors, and approximate quantity.
2. Analyze the pots in the image, including their shapes, sizes, materials, and any patterns or decorations.
3. Describe the grass area, including its condition (e.g., lush, patchy, dry) and approximate dimensions.
4. Provide dimensions or spatial details for the overall backyard, including estimates of the area covered by grass, flowers, and other visible elements.
5. Offer an overall description of the backyard, highlighting key features, aesthetics, and any unique aspects of the design or layout.
"""


@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only JPEG and PNG are allowed.")

    try:
        # Encode the image
        base64_image = encode_image(file)

        # Interact with OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
        )

        # Return response
        return JSONResponse(content={"response": response.choices[0].message.content})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")
