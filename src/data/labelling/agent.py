import os
import asyncio
from typing import Union, List
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

# Set up logging

load_dotenv()

# Global list of API keys for rotation
_api_keys = os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")).split(",")
_api_keys = [k.strip() for k in _api_keys if k.strip()]
_current_key_idx = 0

async def get_lru_api_key():
    """Returns the next API key in the list (Round Robin/LRU)."""
    global _current_key_idx
    if not _api_keys:
        raise ValueError("No GEMINI_API_KEYS found in environment.")

    key = _api_keys[_current_key_idx]
    _current_key_idx = (_current_key_idx + 1) % len(_api_keys)
    return key

async def GeminiAgent(
    model: str,
    contents: Union[types.ContentListUnion, types.ContentListUnionDict],
    config: types.GenerateContentConfigOrDict,
    retry_delay: float = 1.0,
    max_retries: int = 5 # Adjusted based on needs
):
    delay = retry_delay
    retry_count = 0

    while True:
        # Lấy key LRU
        api_key = await get_lru_api_key()
        client = genai.Client(api_key=api_key)

        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

            if not response.text or not response.text.strip():
                # Some safety checks for response validity
                if hasattr(response, 'candidates') and response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                    print(f"[GeminiAgent] Empty response text. Finish reason: {finish_reason}")
                raise ValueError("GeminiAgent: response.text is không hợp lệ")

            return response

        except errors.APIError as e:
            code = e.code
            msg = e.message or ""
            print(f"[GeminiAgent] APIError (Code {code}): {msg}")

            if code == 429:
                # Chỉ cần log và loop tiếp để lấy key khác
                print(f"[GeminiAgent] Key `{api_key[:8]}...` got rate-limited, rotating to next key.")
                # reset delay cho lần dùng key mới
                delay = retry_delay
                continue

            if str(code).startswith("5"):
                print(f"[GeminiAgent] Server error {code}, waiting {delay:.1f}s then retrying...")
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, 60)
                continue

            # Các lỗi 4xx khác – retry tối đa
            retry_count += 1
            if retry_count > max_retries:
                print(f"[GeminiAgent] Client error {code} exceeded {max_retries} retries, stopping.")
                raise
            print(f"[GeminiAgent] Client error {code}, retrying {retry_count}/{max_retries} after {delay:.1f}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 60)

        except Exception as e:
            print(f"[GeminiAgent] Unknown error: {type(e).__name__} – {e}")
            raise

async def generate(image_bytes: bytes, prompt: str = "Please describe this image in detail."):
    # Note: Ensure this model name is available in your region/project
    model = "gemini-3-flash-preview" # experiment version verified, do not change

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=image_bytes,
                ),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL",
        ),
        media_resolution="MEDIA_RESOLUTION_HIGH",
    )

    response = await GeminiAgent(
        model=model,
        contents=contents,
        config=generate_content_config
    )
    return response.text

if __name__ == "__main__":
    # Ensure you have an image.jpg or change this path
    image_path = "./sample/1.jpg"
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        # Use asyncio.run to execute the async function
        result = asyncio.run(generate(image_bytes))
        print(result)
    else:
        print(f"Error: {image_path} not found.")
