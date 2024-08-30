from typing import List
from tqdm import tqdm
import logging
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image


logger = logging.getLogger(__name__)


class MiniCPMHelper:
    def __init__(self, model_name_or_path: str, device: str):
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        self.model.eval()

    def __call__(
        self,
        prompts: List[str],
        image_paths: List[str] = None,
        show_progress_bar: bool = True,
        **generation_kwargs
    ):

        completions = list()
        for idx, prompt in tqdm(
            enumerate(prompts),
            desc="Prompt",
            total=len(prompts),
            disable=not show_progress_bar,
        ):
            image = None
            if image_paths:
                image_path = image_paths[idx]
                image = Image.open(image_path).convert("RGB")

            msgs = [{"role": "user", "content": prompt}]

            res = self.model.chat(
                image=image,
                msgs=msgs,
                tokenizer=self.tokenizer,
                **generation_kwargs,
            )

            completions.append(res)

        return completions
