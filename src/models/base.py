from abc import ABC, abstractmethod
from typing import List
from tqdm import tqdm


class BaseHelper(ABC):

    @abstractmethod
    def _forward(self):
        pass

    def __call__(
        self,
        prompts: List[str],
        image_paths: List[str] = None,
        show_progress_bar: bool = True,
        log_output_every: int = -1,
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

                if log_output_every > 0 and idx % log_output_every == 0:
                    print(f"Prompt: {prompt}")
                    print(f"Image: {image_path}")
                    print(f"Completion: {res}")
                    print("#" * 50)
            except Exception as e:
                print("Failed", idx, prompt, image_path)
                raise e

        return completions
