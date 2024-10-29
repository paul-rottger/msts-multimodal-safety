"""Generic top-level import module."""

from .xgen import XGenHelper
from .minicpm import MiniCPMHelper
from .internvl2 import InternVL2Helper
from .internlm_xcomposer import InternLMXComposerHelper

try:
    from .claude import ClaudeHelper
except ImportError:
    print("Claude model not available.")

try:
    from .gemini import GeminiHelper
except ImportError:
    print("Gemini model not available.")

try:
    from .gpt4 import GPT4VisionHelper
except ImportError:
    print("GPT4Vision model not available.")


# Conditional imports do to interoperatibility issues with HuggingFace versions
try:
    from .idefics import IdeficsHelper
except ImportError:
    print("Idefics model not available.")

try:
    from .qwel2_vl import Qwen2VLHelper
except ImportError:
    print("Qwen2VL model not available.")

try:
    from .cambrian import CambrianHelper
except ImportError:
    print("Cambrain model not available.")
