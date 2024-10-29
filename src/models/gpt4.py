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
import json
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class GPT4VisionHelper:
    def __init__(self, model_name: str):
        self.model_name = model_name

        try:
            self.client = OpenAI()
        except Exception as e:
            logger.error(f"Error connecting to OpenAI API: {e}")

    def __call__(
        self,
        prompts: List[str],
        image_paths: List[str] = None,
        max_new_tokens: int = 256,
        show_progress_bar: bool = True,
    ):
        """Generate GPT4 completions using local images and prompts."""

        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(payload):
            return self.client.chat.completions.create(**payload)

        completions = list()
        for idx, prompt in tqdm(
            enumerate(prompts),
            desc="Item",
            total=len(prompts),
            disable=not show_progress_bar,
        ):
            content = [
                {"type": "text", "text": prompt},
            ]

            if image_paths:
                base64_image = encode_image(image_paths[idx])
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "max_tokens": max_new_tokens,
            }

            try:
                chat_response = completion_with_backoff(payload)
                response = chat_response.choices[0].message.content
            except RetryError:
                logger.warning(f"Retrying with OPENAI API failed.")
                logger.warning(f"Failing row {idx}, prompt: {prompt}")
                response = "FAILED"

            completions.append(response)

        return completions

    def get_batch_lines(self, case_ids, prompts, image_paths, **generation_kwargs):
        """Generate GPT4 completions using local images and prompts."""

        print("Number of propmts: ", len(prompts))

        json_lines = []
        for idx, (case_id, prompt) in enumerate(zip(case_ids, prompts)):
            content = [
                {"type": "text", "text": prompt},
            ]

            if image_paths:
                base64_image = encode_image(image_paths[idx])
                image_extension = image_paths[idx].split(".")[-1].lower()
                if image_extension in ["jpg", "jpeg"]:
                    mime_type = "image/jpeg"
                elif image_extension == "png":
                    mime_type = "image/png"
                else:
                    raise ValueError(f"Unsupported image format: {image_extension}")

                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    }
                )

            payload = {
                "custom_id": case_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    "max_tokens": generation_kwargs["max_new_tokens"],
                },
            }

            # json_lines.append(json.dumps(payload))
            json_line_str = (
                json.dumps(payload, ensure_ascii=False).encode("utf8").decode()
            )
            json_lines.append(json_line_str)

        return json_lines
