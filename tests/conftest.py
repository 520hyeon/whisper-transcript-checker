from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture
def samples() -> Path:
    return SAMPLES


_WORDS = ["사과", "바다", "구름", "연필", "기차", "나무", "의자", "편지", "우산", "시계"]


def seg(idx, start, end, text=None, **metrics):
    """테스트용 세그먼트. text를 주지 않으면 idx마다 다른 평범한 문장 (앞뒤가 같으면 dup으로 잡히므로)."""
    if text is None:
        text = f"{_WORDS[idx % len(_WORDS)]} 이야기를 짧게 했습니다"
    return {"idx": idx, "start": float(start), "end": float(end), "text": text, **metrics}
