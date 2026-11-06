"""faster-whisper 세그먼트에서 환각·누락·불안정 구간을 찾는 도구."""
__version__ = "0.1.0"

from .detect import DEFAULTS, CheckResult, check  # noqa: E402
from .loader import InputError, Transcript, load, parse  # noqa: E402

__all__ = ["DEFAULTS", "CheckResult", "InputError", "Transcript", "check", "load", "parse", "__version__"]
