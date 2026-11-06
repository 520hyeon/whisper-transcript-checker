import pytest

from transcript_checker.detect import (counting_run, dup_flags, hallucination_reasons, is_english_only,
                                       longest_common_run, merged_settings, number_reasons, repeated_run,
                                       tokens, unstable_reasons)

S = merged_settings()


def test_tokens_lowercases_and_masks_digits():
    assert tokens("Hello, 세계 2024년 A1!") == ["hello", "세계", "#년", "a#"]
    assert tokens(None) == []


def test_repeated_run_finds_longest_repeated_ngram():
    assert repeated_run("a b a b a b c".split()) == 6
    assert repeated_run("가 가 가 가".split()) == 4
    assert repeated_run("가 나 다 라".split()) == 0


def test_repeated_run_ignores_number_only_units():
    assert repeated_run(["#"] * 10) == 0


def test_counting_run_breaks_on_words():
    assert counting_run("1, 2, 3, 4") == 4
    assert counting_run("1, 2, 다음 3, 4, 5") == 3
    assert counting_run("2, 8, 7, 1") == 1


def test_number_reasons():
    counting = ", ".join(str(n) for n in range(1, 15))
    assert set(number_reasons(counting, S)) == {"numbers", "counting"}
    assert number_reasons("배열 2, 8, 7, 1, 3, 5, 6, 4를 정렬합니다", S) == []
    many = " ".join(f"항목 {n}" for n in range(1, 10))  # 같은 형태(# 숫자) 토큰과 '항목'이 반복
    assert "same_token" in number_reasons(many, S)


def test_longest_common_run():
    assert longest_common_run("a b c d".split(), "x b c d y".split()) == 3
    assert longest_common_run([], ["a"]) == 0


def test_is_english_only():
    assert is_english_only("This is clearly an English sentence", 4)
    assert not is_english_only("이건 chain checker 도구입니다", 1)
    assert not is_english_only("x y z n", 4)  # 한 글자 기호는 단어로 세지 않음


@pytest.mark.parametrize("text, reason", [
    ("물을 주세요 물을 주세요 물을 주세요 물을 주세요", "repeat"),
    ("시청해 주셔서 감사합니다", "phrase"),
    ("시 청 해 주 셔 서 감사합니다", "phrase"),  # 띄어쓰기가 달라도 잡음
])
def test_hallucination_text_reasons(text, reason):
    assert reason in hallucination_reasons({"text": text}, S, "ko")


def test_short_repetition_is_not_flagged():
    assert hallucination_reasons({"text": "네 네 네"}, S, "ko") == []  # 반복 구간이 4단어 미만


def test_english_only_depends_on_language():
    seg = {"text": "Water the plants once every two days."}
    assert "english" in hallucination_reasons(seg, S, "ko")
    assert "english" not in hallucination_reasons(seg, S, "en")
    assert "english" not in hallucination_reasons(seg, S, None)


def test_compression_ratio_threshold():
    assert "compression" in hallucination_reasons({"text": "아무 말", "compression_ratio": 2.5}, S)
    assert "compression" not in hallucination_reasons({"text": "아무 말", "compression_ratio": 2.4}, S)


def test_prompt_leak_needs_long_overlap():
    prompt = ["상자 텃밭, 배수층, 순지르기, 웃거름"]
    assert "prompt" in hallucination_reasons({"text": "상자 텃밭 배수층 순지르기"}, S, prompt_texts=prompt)
    # 프롬프트 용어 하나를 자연스럽게 말한 문장은 통과
    assert "prompt" not in hallucination_reasons(
        {"text": "오늘은 배수층을 만드는 방법을 차근차근 이야기해 보겠습니다"}, S, prompt_texts=prompt)


def test_unstable_reasons():
    assert unstable_reasons({"text": "뭔가", "avg_logprob": -1.2}, S) == ["low_logprob"]
    assert unstable_reasons({"text": "뭔가", "no_speech_prob": 0.9}, S) == ["no_speech"]
    assert unstable_reasons({"text": "", "no_speech_prob": 0.9}, S) == []  # 텍스트가 없으면 정상
    assert unstable_reasons({"text": "뭔가"}, S) == []  # 지표가 없으면 판단하지 않음


def test_dup_flags_marks_both_neighbours_and_skips_short_text():
    segs = [{"text": "같은 문장이 두 번 나옵니다"}, {"text": "같은 문장이 두 번 나옵니다."}, {"text": "다른 문장"}]
    assert dup_flags(segs, S) == {0, 1}
    assert dup_flags([{"text": "네"}, {"text": "네"}], S) == set()
