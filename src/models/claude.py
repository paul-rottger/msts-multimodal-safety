import logging
import base64
from .base import BaseHelper

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ClaudeHelper(BaseHelper):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = anthropic.Anthropic(max_retries=10, timeout=60)

    def _forward(self, prompt, image_path, **generation_kwargs):
        """Process a single image and prompt."""

        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encode_image(image_path),
                },
            },
        ]

        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=generation_kwargs.get("max_new_tokens", 256),
            messages=[{"role": "user", "content": content}],
        )
        return message.content[0].text
