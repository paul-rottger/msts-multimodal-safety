from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List
import torch


# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


class Qwen2VLHelper:
    def __init__(self, model_name_or_path: str, device: str):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

    @torch.inference_mode()
    def _forward(self, prompt, image_path, **generation_kwargs):
        """Process a single image and prompt."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return response

    def __call__(
        self,
        prompts: List[str],
        image_paths: List[str] = None,
        show_progress_bar: bool = True,
        **generation_kwargs,
    ):
        """Generate completions using local images and prompts."""
        assert len(prompts) == len(image_paths)

        completions = list()
        for idx, (prompt, image_path) in tqdm(
            enumerate(zip(prompts, image_paths)),
            desc="Item:",
            disable=not show_progress_bar,
            total=len(prompts),
        ):
            try:
                res = self._forward(prompt, image_path, **generation_kwargs)
                completions.append(res)
            except Exception as e:
                print("Failed", idx, prompt, image_path)
                raise e

        return completions
