"""faster-whisper 세그먼트를 JSON 또는 JSONL에서 읽는다.

JSON
  - 세그먼트 배열: [{"start": 0.0, "end": 2.5, "text": "..."}, ...]
  - 객체: {"language": "ko", "duration": 612.3, "segments": [...]}
    language·duration은 최상위나 "info" 안에 있어도 된다 (faster-whisper의 TranscriptionInfo).
JSONL
  - 한 줄에 세그먼트 하나. 빈 줄은 건너뛴다.
  - start가 없고 language나 duration이 있는 줄은 메타데이터로 읽는다.

세그먼트 필드: start, end(초, 필수), text(문자열), avg_logprob, no_speech_prob, compression_ratio(선택).
그 밖의 필드(words, tokens 등)는 무시한다. 세그먼트 번호(idx)는 파일 안 순서(0부터)다.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

METRICS = ("avg_logprob", "no_speech_prob", "compression_ratio")
FORMATS = ("auto", "json", "jsonl")


class InputError(ValueError):
    """입력 파일 형식이 잘못됨."""


@dataclass
class Transcript:
    segments: List[Dict[str, Any]]
    language: Optional[str] = None
    duration: Optional[float] = None
    source: str = "-"


def _number(value, where, name, required):
    if value is None:
        if required:
            raise InputError(f"{where}: '{name}' 필드가 없습니다")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
        raise InputError(f"{where}: '{name}'는 숫자여야 합니다 (받은 값: {value!r})")
    return float(value)


def parse_segment(obj: Any, idx: int, where: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise InputError(f"{where}: 세그먼트는 JSON 객체여야 합니다")
    start = _number(obj.get("start"), where, "start", True)
    end = _number(obj.get("end"), where, "end", True)
    if end < start:
        raise InputError(f"{where}: end({end})가 start({start})보다 앞입니다")
    text = obj.get("text", "")
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise InputError(f"{where}: 'text'는 문자열이어야 합니다")
    seg = {"idx": idx, "start": start, "end": end, "text": text.strip()}
    for k in METRICS:
        v = _number(obj.get(k), where, k, False)
        if v is not None:
            seg[k] = v
    return seg


def _meta(obj: Dict[str, Any], where: str):
    info = obj.get("info") if isinstance(obj.get("info"), dict) else {}
    language = obj.get("language", info.get("language"))
    duration = obj.get("duration", info.get("duration"))
    if language is not None and not isinstance(language, str):
        raise InputError(f"{where}: 'language'는 문자열이어야 합니다")
    return language, _number(duration, where, "duration", False)


def parse_json(text: str) -> Transcript:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise InputError(f"JSON을 읽을 수 없습니다 ({e.lineno}번째 줄 {e.colno}번째 글자: {e.msg})") from None
    language = duration = None
    if isinstance(data, dict):
        if "segments" not in data:
            raise InputError("JSON 객체에 'segments' 배열이 없습니다")
        language, duration = _meta(data, "최상위 객체")
        data = data["segments"]
    if not isinstance(data, list):
        raise InputError("세그먼트 배열을 찾을 수 없습니다 (배열이나 {\"segments\": [...]} 형식이어야 합니다)")
    segs = [parse_segment(obj, i, f"segments[{i}]") for i, obj in enumerate(data)]
    return Transcript(segs, language, duration)


def parse_jsonl(text: str) -> Transcript:
    segs: List[Dict[str, Any]] = []
    language = duration = None
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        where = f"{lineno}번째 줄"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise InputError(f"{where}: JSON을 읽을 수 없습니다 ({e.msg})") from None
        if isinstance(obj, dict) and "start" not in obj and ("language" in obj or "duration" in obj or "info" in obj):
            lang, dur = _meta(obj, where)
            language = lang if lang is not None else language
            duration = dur if dur is not None else duration
            continue
        segs.append(parse_segment(obj, len(segs), where))
    return Transcript(segs, language, duration)


def parse(text: str, fmt: str = "auto") -> Transcript:
    if fmt not in FORMATS:
        raise InputError(f"알 수 없는 형식: {fmt} ({', '.join(FORMATS)} 중 하나)")
    text = text.lstrip("﻿")
    if not text.strip():
        raise InputError("입력이 비어 있습니다")
    if fmt == "json":
        return parse_json(text)
    if fmt == "jsonl":
        return parse_jsonl(text)
    # auto: 파일 내용 전체가 JSON 값 하나이고 배열이나 {"segments": ...}면 JSON, 아니면 JSONL
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return parse_jsonl(text)
    if isinstance(data, list) or (isinstance(data, dict) and "segments" in data):
        return parse_json(text)
    return parse_jsonl(text)  # 세그먼트 한 줄짜리 JSONL 등


def load(path: str, fmt: str = "auto") -> Transcript:
    """파일(또는 "-"이면 표준 입력)에서 세그먼트를 읽는다."""
    if path == "-":
        try:
            text = sys.stdin.buffer.read().decode("utf-8")
        except UnicodeDecodeError:
            raise InputError("표준 입력을 UTF-8로 읽을 수 없습니다") from None
        source = "stdin"
    else:
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise InputError(f"파일이 없습니다: {p.name}") from None
        except OSError as e:
            raise InputError(f"파일을 열 수 없습니다: {p.name} ({e.strerror})") from None
        except UnicodeDecodeError:
            raise InputError(f"UTF-8로 읽을 수 없습니다: {p.name}") from None
        source = p.name  # 리포트에 개인 경로가 남지 않도록 파일 이름만
    t = parse(text, fmt)
    t.source = source
    return t
