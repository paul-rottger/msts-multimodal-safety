import base64
from typing import List
from tqdm import tqdm
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError,
)
import google.generativeai as genai
import PIL.Image
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GeminiHelper:
    def __init__(self, model_name: str):
        self.model_name = model_name
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model_name=model_name)

    def __call__(
        self,
        prompts: List[str],
        image_paths: List[str] = None,
        max_new_tokens: int = 256,
        show_progress_bar: bool = True,
    ):
        """Generate Gemini completions using local images and prompts."""

        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(prompt, img):
            response = self.model.generate_content(
                [prompt, img],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_new_tokens
                ),
            )
            text = (
                response.text
                if hasattr(response, "text")
                else str(response.candidates[0].finish_reason)
            )
            return text

        completions = list()
        for idx, prompt in tqdm(
            enumerate(prompts),
            desc="Item",
            total=len(prompts),
            disable=not show_progress_bar,
        ):
            img = PIL.Image.open(image_paths[idx])
            try:
                response = completion_with_backoff(prompt, img)
            except RetryError as e:
                logger.warning(f"Retrying with Gemini API failed.")
                logger.warning(f"Failing row {idx}, prompt: {prompt}")
                response = "FAILED"

            completions.append(response)

        return completions
