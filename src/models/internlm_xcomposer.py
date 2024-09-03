import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from tqdm import tqdm


class InternLMXComposerHelper:
    def __init__(self, model_name_or_path: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )

        self.model.tokenizer = self.tokenizer

    @torch.inference_mode()
    def _forward(self, prompt, image_path, **generation_kwargs):
        """Process a single image and prompt."""
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            response, _ = self.model.chat(
                self.tokenizer, prompt, [image_path], **generation_kwargs
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
