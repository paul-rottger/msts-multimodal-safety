from typing import List
from tqdm import tqdm
import logging
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image

from .base import BaseHelper

logger = logging.getLogger(__name__)


class MiniCPMHelper(BaseHelper):
    def __init__(self, model_name_or_path: str, device: str):
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        self.model.eval()

    @torch.inference_mode()
    def _forward(self, prompt, image_path: str=None, **generation_kwargs):
        
        image = None
        if image_path:
            image = Image.open(image_path)
            
        msgs = [{"role": "user", "content": prompt}]
        res = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            **generation_kwargs,
        )
        return res
