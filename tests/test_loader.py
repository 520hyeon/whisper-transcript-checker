import json

import pytest

from transcript_checker.loader import InputError, load, parse


def test_json_array():
    t = parse('[{"start": 0, "end": 1.5, "text": " 안녕 "}]')
    assert t.segments == [{"idx": 0, "start": 0.0, "end": 1.5, "text": "안녕"}]
    assert t.language is None and t.duration is None


def test_json_object_with_info_and_metrics():
    data = {"info": {"language": "ko", "duration": 12.5},
            "segments": [{"id": 1, "start": 0, "end": 2, "text": "가", "avg_logprob": -0.3,
                          "no_speech_prob": 0.1, "compression_ratio": 1.2, "words": [{"word": "가"}]}]}
    t = parse(json.dumps(data))
    assert t.language == "ko" and t.duration == 12.5
    assert t.segments[0] == {"idx": 0, "start": 0.0, "end": 2.0, "text": "가",
                             "avg_logprob": -0.3, "no_speech_prob": 0.1, "compression_ratio": 1.2}


def test_top_level_metadata_wins_over_info():
    t = parse(json.dumps({"language": "en", "info": {"language": "ko"}, "segments": []}))
    assert t.language == "en"


def test_jsonl_with_metadata_line_and_blank_lines():
    text = '{"language": "ko", "duration": 30}\n\n{"start": 0, "end": 1, "text": "a"}\n{"start": 1, "end": 2}\n'
    t = parse(text)
    assert t.language == "ko" and t.duration == 30.0
    assert [s["idx"] for s in t.segments] == [0, 1]
    assert t.segments[1]["text"] == ""


def test_auto_detects_single_line_jsonl():
    t = parse('{"start": 0, "end": 1, "text": "한 줄"}')
    assert len(t.segments) == 1


def test_forced_format():
    with pytest.raises(InputError):
        parse('{"start": 0, "end": 1}\n{"start": 1, "end": 2}', "json")
    assert len(parse('[{"start": 0, "end": 1}]', "json").segments) == 1


def test_bom_is_ignored():
    assert len(parse('﻿[{"start": 0, "end": 1}]').segments) == 1


@pytest.mark.parametrize("text, message", [
    ("", "비어"),
    ('{"start": 0, "end": 1}\n{oops}', "2번째 줄"),
    ('[{"end": 1}]', "'start' 필드가 없습니다"),
    ('[{"start": "0", "end": 1}]', "숫자여야"),
    ('[{"start": true, "end": 1}]', "숫자여야"),
    ('[{"start": 3, "end": 1}]', "end(1.0)가 start(3.0)보다 앞"),
    ('[{"start": 0, "end": 1, "text": 5}]', "'text'는 문자열"),
    ('[{"start": 0, "end": 1, "avg_logprob": "low"}]', "'avg_logprob'"),
    ('{"segments": {"start": 0}}', "배열"),
    ('{"language": "ko"}\n[1, 2]', "2번째 줄"),
    ('{"segments": [], "duration": "long"}', "'duration'"),
])
def test_invalid_input_messages(text, message):
    with pytest.raises(InputError, match=message.replace("(", r"\(").replace(")", r"\)")):
        parse(text)


def test_load_reports_only_file_name(tmp_path):
    p = tmp_path / "segments.jsonl"
    p.write_text('{"start": 0, "end": 1, "text": "가"}\n', encoding="utf-8")
    t = load(str(p))
    assert t.source == "segments.jsonl"
    with pytest.raises(InputError) as e:
        load(str(tmp_path / "missing.json"))
    assert str(tmp_path) not in str(e.value)


def test_load_rejects_non_utf8(tmp_path):
    p = tmp_path / "bad.json"
    p.write_bytes('[{"start": 0, "end": 1, "text": "가"}]'.encode("cp949"))
    with pytest.raises(InputError, match="UTF-8"):
        load(str(p))
