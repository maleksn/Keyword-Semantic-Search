# File: ./cli/describe_image_cli.py
import argparse
import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI


def get_llm_client() -> OpenAI:
    """Create an OpenAI client pointed at OpenRouter."""
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable not set"
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal query rewriting using image + text"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query to rewrite based on the image",
    )

    args = parser.parse_args()

    # Validate image file exists
    if not os.path.isfile(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    # Determine MIME type
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    # Read image in binary mode
    with open(args.image, "rb") as f:
        img_bytes = f.read()

    # Encode image as base64 data URL
    data_url = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"

    # System prompt for multimodal query rewriting
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    # Build multimodal message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]

    # Send request to OpenRouter
    client = get_llm_client()

    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
        )

        content = response.choices[0].message.content
        print(f"Rewritten query: {content.strip()}")

        if response.usage is not None:
            print(f"Total tokens:    {response.usage.total_tokens}")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()