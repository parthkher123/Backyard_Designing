from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import torch
from huggingface_hub import login
from diffusers import StableDiffusionDepth2ImgPipeline

# FastAPI app
app = FastAPI()

# Authenticate with Hugging Face
login(token="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")

# Ensure CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained StableDiffusion model
pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
).to(device)

details = "Based on the provided image of the backyard, here’s a detailed analysis:\n\n1. **Visible Flowers**: \n   - There are limited visible flowers in the image. Most of the visible plants appear to be shrubs and ornamental trees rather than distinct flowering plants. Thus, specific types and quantities of flowers are not identifiable.\n\n2. **Pots**: \n   - The image does not show any distinct pots or containers. The focus is primarily on the landscape with grass and planted areas.\n\n3. **Grass Area**: \n   - The grass is lush and well-maintained, indicating regular care. It appears to be a uniform green color with no visible patches. The approximate dimensions of the grass area can be estimated to cover the majority of the visible lawn, likely around 75% of the total yard space.\n\n4. **Dimensions and Spatial Details**: \n   - The overall backyard seems quite spacious, though exact dimensions cannot be determined from the image. Based on the visual elements, the area covered by grass could be estimated at roughly 30-50 feet in width and 50-100 feet in length, totaling an approximate grass area of 1500-5000 square feet. The flower beds and features along the edges occupy a smaller portion of the overall area.\n\n5. **Overall Description**: \n   - The backyard showcases a well-kept, inviting landscape with a primary focus on a lush, green lawn. The surrounding trees and shrubs provide a natural boundary, enhancing privacy and creating a serene atmosphere. The aesthetic is calm and harmonious, with the greenery complementing the wooden fence that adds a natural texture. The design emphasizes open space, accessibility, and a peaceful environment, ideal for family gatherings or relaxation.\n\nThis analysis highlights the overall beauty and tranquility of the backyard, focusing on maintenance and layout rather than specific decorative features."
@app.post("/generate-image/")
async def generate_image(file: UploadFile = File(...), details: str = ""):
    # Read the input image
    image_data = await file.read()
    init_image = Image.open(BytesIO(image_data))

    # Set the prompt with the details
    positive_prompt = (
        "A serene backyard with hydrangeas, azaleas, and dogwoods in full bloom. "
        "Soft, pastel flowers provide a calming retreat under partial shade. "
        "Vibrant greenery, peaceful atmosphere, warm sunlight filtering through the trees."
    )
    negative_prompt = (
        "blurry, deformed, unnatural, cartoonish, overexposed, dull colors, artificial look, "
        "low-resolution, overly saturated, harsh lighting, distorted flowers, unrealistic textures"
    )

    # Generate the image
    generated_image = pipe(prompt=positive_prompt, image=init_image, negative_prompt=negative_prompt, strength=0.7).images[0]

    # Save the image to BytesIO object for return in response
    img_byte_arr = BytesIO()
    generated_image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    return {"filename": "generated_image.jpg", "image": img_byte_arr.getvalue()}

