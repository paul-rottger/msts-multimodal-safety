"""Generic top-level import module."""

from .gpt4 import GPT4VisionHelper
from .gemini import GeminiHelper
from .xgen import XGenHelper
from .minicpm import MiniCPMHelper
from .internvl2 import InternVL2Helper
from .internlm_xcomposer import InternLMXComposerHelper
from .claude import ClaudeHelper

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
