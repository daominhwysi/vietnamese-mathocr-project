import os
import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
async def generate(image_bytes: bytes):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # Note: Ensure this model name is available in your region/project
    model = "gemini-3-flash-preview"

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=image_bytes,
                ),
                types.Part.from_text(text="Please describe this image in detail."),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
        ),
        # Note: media_resolution usage depends on specific model support
    )

    # Added 'await' here
    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config
    )
    return response.text

if __name__ == "__main__":
    # Ensure you have an image.jpg or change this path
    image_path = "./sample/1.jpg"
    if os.path.exists(image_path):
        image_bytes = open(image_path, "rb").read()
        # Use asyncio.run to execute the async function
        result = asyncio.run(generate(image_bytes))
        print(result)
    else:
        print(f"Error: {image_path} not found.")
