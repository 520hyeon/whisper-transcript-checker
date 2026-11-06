"""세그먼트 텍스트와 Whisper 지표만으로 환각·누락·불안정 구간을 찾는다 (오디오 없이 동작).

분류와 이유 코드
  hallucination (환각 의심)
    repeat       같은 단어·구절이 연속 3번 이상 반복되고 문장의 절반 이상을 차지
    dup          바로 앞이나 뒤 세그먼트와 같은 문장
    prompt       initial_prompt·hotwords 문구가 그대로 나옴
    english      language=ko인데 한글 없이 영어 단어만 4개 이상 (번역 출력 의심)
    compression  compression_ratio 과다
    phrase       "시청해주셔서 감사합니다" 같은 알려진 환각 문구
    numbers      숫자가 대부분이고 단어가 거의 없음
    counting     1씩 늘어나는 숫자 세기가 길게 이어짐
    same_token   길이에 비해 같은 형태의 토큰(숫자는 #)이 지나치게 많음
  omission (누락 의심)
    gap          세그먼트 사이(또는 처음·끝) 간격이 기준 이상
    sparse       긴 세그먼트에 글자가 너무 적음
  unstable (인식 불안정)
    low_logprob  avg_logprob가 기준보다 낮음
    no_speech    no_speech_prob가 높은데 텍스트가 있음

가까운 의심 항목은 하나의 구간(region)으로 합친다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

DEFAULTS: Dict[str, Any] = {
    # 환각
    "repeat_min_times": 3,             # 같은 단어·구절이 연속 이만큼 반복되고
    "repeat_min_words": 4,             # 반복 구간이 이 단어 수 이상이며
    "repeat_min_ratio": 0.5,           # 문장 전체 단어의 이 비율 이상이면 환각 의심
    "dup_min_words": 3,                # 앞뒤와 같은 문장이 이 단어 수 이상이거나
    "dup_min_chars": 8,                # 이 글자 수 이상이면 dup
    "prompt_min_words": 3,             # 프롬프트와 연속으로 겹치는 단어가 이만큼 이상이고
    "prompt_min_ratio": 0.6,           # 문장의 이 비율 이상이면 프롬프트 누출 의심
    "english_min_words": 4,            # language=ko인데 한글 없이 영어 단어가 이만큼 이상이면 번역 출력 의심
    "hallucination_compression": 2.4,  # compression_ratio가 이보다 높으면 환각 의심
    "hallucination_phrases": ["시청해주셔서 감사합니다", "시청해 주셔서 감사합니다", "구독과 좋아요",
                              "구독 좋아요", "좋아요와 구독", "MBC 뉴스", "자막 제공", "Thank you for watching"],
    "numbers_min_count": 12,           # 숫자 토큰이 이만큼 이상이고
    "numbers_min_ratio": 0.8,          # 전체 토큰의 이 비율 이상이면 숫자 나열 의심
    "counting_min_run": 10,            # 1씩 늘어나는 숫자가 이만큼 연속이면 숫자 세기 의심
    "same_token_min_count": 16,        # 토큰이 이만큼 이상인 세그먼트에서
    "same_token_ratio": 0.5,           # 같은 형태의 토큰이 이 비율 이상이면 환각 의심
    # 누락
    "gap_min_seconds": 4.0,            # 세그먼트 사이 간격이 이 이상이면 누락 의심
    "sparse_min_seconds": 8.0,         # 이보다 긴 세그먼트가
    "sparse_max_cps": 1.2,             # 초당 이 글자 수(공백 제외) 미만이면 누락 의심
    # 불안정
    "low_logprob": -1.0,               # avg_logprob가 이보다 낮으면 불안정
    "no_speech_prob": 0.6,             # no_speech_prob가 이 이상인데 텍스트가 있으면 불안정
    # 구간 합치기
    "region_join_seconds": 1.0,        # 이보다 가까운 의심 항목은 하나의 구간으로 합침
}

CATEGORIES = ("hallucination", "omission", "unstable")
CATEGORY_TEXT = {"hallucination": "환각 의심", "omission": "누락 의심", "unstable": "인식 불안정"}
HALLUCINATION = ("repeat", "dup", "prompt", "english", "compression", "phrase", "numbers", "counting", "same_token")
OMISSION = ("gap", "sparse")
UNSTABLE = ("low_logprob", "no_speech")
CATEGORY_OF = {**{r: "hallucination" for r in HALLUCINATION},
               **{r: "omission" for r in OMISSION},
               **{r: "unstable" for r in UNSTABLE}}
REASON_TEXT = {"repeat": "같은 말 반복", "dup": "앞뒤 문장과 같음", "prompt": "프롬프트 누출",
               "english": "영어로만 출력(번역 의심)", "compression": "압축률 과다", "phrase": "알려진 환각 문구",
               "numbers": "숫자 나열", "counting": "숫자 세기", "same_token": "같은 형태 토큰 과다",
               "gap": "전사 없는 간격", "sparse": "길이에 비해 글자가 적음",
               "low_logprob": "낮은 avg_logprob", "no_speech": "높은 no_speech_prob"}

_HANGUL = re.compile(r"[가-힣]")
_TOKEN = re.compile(r"[0-9A-Za-z가-힣#]+")
_LATIN_WORD = re.compile(r"^(?=.*[a-z].*[a-z])[a-z#]+$")  # 영어 단어 (알파벳 2자 이상, "x"·"n" 같은 기호는 제외)
_SPACE = re.compile(r"\s+")


def merged_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    s = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    s.update(overrides or {})
    return s


# ---------- 텍스트 검사 ----------

def tokens(text: Optional[str]) -> List[str]:
    """비교용 단어 목록: 소문자, 숫자는 #, 문장부호 제거."""
    return _TOKEN.findall(re.sub(r"\d+", "#", (text or "").lower()))


def norm(text: Optional[str]) -> str:
    return " ".join(tokens(text))


def char_count(text: Optional[str]) -> int:
    return len(_SPACE.sub("", text or ""))


def repeated_run(toks: Sequence[str], min_times: int = 3, max_n: int = 8) -> int:
    """같은 n-gram(n ≤ max_n)이 연속 min_times번 이상 반복되는 가장 긴 구간의 단어 수.
    숫자만 있는 단위("2, 8, 7, 1")는 제외."""
    best, total = 0, len(toks)
    for n in range(1, max_n + 1):
        i = 0
        while i + n * min_times <= total:
            unit = list(toks[i:i + n])
            if all(t == "#" for t in unit):
                i += 1
                continue
            k = 1
            while list(toks[i + k * n:i + (k + 1) * n]) == unit:
                k += 1
            if k >= min_times:
                best = max(best, k * n)
                i += k * n
            else:
                i += 1
    return best


def counting_run(text: Optional[str]) -> int:
    """1씩 늘어나는 숫자가 연속으로 나오는 가장 긴 길이 ("1, 2, 3, 4" → 4). 사이에 낱말이 끼면 끊긴다."""
    best = run = 0
    prev = None
    for t in re.findall(r"\d+|[A-Za-z가-힣]{2,}", text or ""):
        if not t.isdigit():
            prev, run = None, 0
            continue
        n = int(t)
        run = run + 1 if prev is not None and n == prev + 1 else 1
        prev = n
        best = max(best, run)
    return best


def number_reasons(text: Optional[str], s: Dict[str, Any]) -> List[str]:
    """숫자 나열 환각 검사: numbers / counting / same_token 중 해당하는 것."""
    text = text or ""
    raw = _TOKEN.findall(text)
    out: List[str] = []
    if not raw:
        return out
    digits = sum(1 for t in re.findall(r"[0-9A-Za-z가-힣]+", text) if re.fullmatch(r"[A-Za-z]?\d+", t))
    if digits >= s["numbers_min_count"] and digits >= len(raw) * s["numbers_min_ratio"]:
        out.append("numbers")
    if counting_run(text) >= s["counting_min_run"]:
        out.append("counting")
    toks = tokens(text)
    if len(toks) >= s["same_token_min_count"]:
        top = max(toks.count(t) for t in set(toks))
        if top >= len(toks) * s["same_token_ratio"]:
            out.append("same_token")
    return out


def longest_common_run(a: Sequence[str], b: Sequence[str]) -> int:
    """두 단어 목록에 공통으로 연속해서 나오는 가장 긴 구간의 길이."""
    if not a or not b:
        return 0
    prev, best = [0] * (len(b) + 1), 0
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def is_english_only(text: Optional[str], min_words: int) -> bool:
    toks = tokens(text)
    return not _HANGUL.search(text or "") and sum(1 for t in toks if _LATIN_WORD.match(t)) >= min_words


def hallucination_reasons(seg: Dict[str, Any], s: Dict[str, Any], language: Optional[str] = None,
                          prompt_texts: Iterable[str] = ()) -> List[str]:
    """세그먼트 하나만 보고 알 수 있는 환각 의심 이유 (dup은 dup_flags에서 따로)."""
    text = seg.get("text") or ""
    toks = tokens(text)
    out: List[str] = []
    if toks and repeated_run(toks, s["repeat_min_times"]) >= max(s["repeat_min_words"], len(toks) * s["repeat_min_ratio"]):
        out.append("repeat")
    words = [t for t in toks if t != "#"]  # 숫자 목록이 프롬프트의 숫자와 겹친다고 보지 않도록
    for p in prompt_texts:
        run = longest_common_run(words, [t for t in tokens(p) if t != "#"])
        if run >= s["prompt_min_words"] and run >= len(words) * s["prompt_min_ratio"]:
            out.append("prompt")
            break
    if language == "ko" and is_english_only(text, s["english_min_words"]):
        out.append("english")
    cr = seg.get("compression_ratio")
    if cr is not None and cr > s["hallucination_compression"]:
        out.append("compression")
    flat = _SPACE.sub("", text)
    if any(_SPACE.sub("", ph) in flat for ph in s["hallucination_phrases"] if ph.strip()):
        out.append("phrase")
    out += number_reasons(text, s)
    return out


def unstable_reasons(seg: Dict[str, Any], s: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    lp = seg.get("avg_logprob")
    if lp is not None and lp < s["low_logprob"]:
        out.append("low_logprob")
    ns = seg.get("no_speech_prob")
    if ns is not None and ns >= s["no_speech_prob"] and char_count(seg.get("text")) > 0:
        out.append("no_speech")
    return out


def dup_flags(segs: Sequence[Dict[str, Any]], s: Dict[str, Any]) -> set:
    """앞뒤 세그먼트와 같은 문장인 세그먼트의 목록 위치."""
    norms = [norm(x.get("text")) for x in segs]
    out = set()
    for k in range(1, len(segs)):
        a = norms[k]
        if a and a == norms[k - 1] and (len(a.split()) >= s["dup_min_words"]
                                        or len(a.replace(" ", "")) >= s["dup_min_chars"]):
            out |= {k - 1, k}
    return out


# ---------- 감지 ----------

@dataclass
class CheckResult:
    segments: List[Dict[str, Any]]           # 시작 시각순
    regions: List[Dict[str, Any]]
    flags: Dict[int, List[str]]              # 세그먼트 idx → 이유 코드
    stats: Dict[str, Any]
    settings: Dict[str, Any]
    language: Optional[str] = None
    duration: Optional[float] = None
    prompt_texts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"language": self.language, "duration": self.duration, "stats": self.stats,
                "regions": self.regions,
                "flagged_segments": [{"idx": i, "reasons": r} for i, r in sorted(self.flags.items())],
                "settings": self.settings}


def check(segments: Sequence[Dict[str, Any]], settings: Optional[Dict[str, Any]] = None,
          language: Optional[str] = None, prompt_texts: Iterable[str] = (),
          duration: Optional[float] = None) -> CheckResult:
    """세그먼트(dict: idx, start, end, text, [avg_logprob, no_speech_prob, compression_ratio]) 검사.

    duration을 주면 마지막 세그먼트 뒤의 간격도 검사한다."""
    s = merged_settings(settings)
    prompts = [p for p in prompt_texts if p and p.strip()]
    segs = sorted(segments, key=lambda x: (x["start"], x["end"]))
    items = []  # (start, end, reason, idx)
    flags: Dict[int, List[str]] = {}

    dups = dup_flags(segs, s)
    for k, seg in enumerate(segs):
        hal = hallucination_reasons(seg, s, language, prompts) + (["dup"] if k in dups else [])
        reasons = hal + unstable_reasons(seg, s)
        dur = seg["end"] - seg["start"]
        if not hal and dur >= s["sparse_min_seconds"] and char_count(seg.get("text")) / dur < s["sparse_max_cps"]:
            reasons.append("sparse")
        for r in reasons:
            items.append((seg["start"], seg["end"], r, seg["idx"]))
        if reasons:
            flags[seg["idx"]] = reasons

    cursor = 0.0  # 처음~첫 세그먼트, 세그먼트 사이, 마지막~끝(duration). 겹치는 세그먼트는 가장 늦은 끝을 기준으로
    for seg in segs:
        if seg["start"] - cursor >= s["gap_min_seconds"]:
            items.append((cursor, seg["start"], "gap", None))
        cursor = max(cursor, seg["end"])
    if duration is not None and duration - cursor >= s["gap_min_seconds"]:
        items.append((cursor, duration, "gap", None))

    regions = _merge(items, s["region_join_seconds"])
    return CheckResult(segments=segs, regions=regions, flags=flags,
                       stats=_stats(segs, regions, flags, duration), settings=s,
                       language=language, duration=duration, prompt_texts=prompts)


def _merge(items, join_seconds) -> List[Dict[str, Any]]:
    items = sorted(items, key=lambda x: (x[0], x[1]))
    regions: List[Dict[str, Any]] = []
    for a, b, r, idx in items:
        cur = regions[-1] if regions else None
        if cur is None or a > cur["end"] + join_seconds:
            cur = {"start": a, "end": b, "categories": [], "reasons": [], "segments": [],
                   "hallucinated": [], "gaps": []}
            regions.append(cur)
        cur["end"] = max(cur["end"], b)
        if r not in cur["reasons"]:
            cur["reasons"].append(r)
        cat = CATEGORY_OF[r]
        if cat not in cur["categories"]:
            cur["categories"].append(cat)
        if r == "gap":
            cur["gaps"].append([round(a, 2), round(b, 2)])
        if idx is not None:
            if idx not in cur["segments"]:
                cur["segments"].append(idx)
            if cat == "hallucination" and idx not in cur["hallucinated"]:
                cur["hallucinated"].append(idx)
    for n, rg in enumerate(regions, 1):
        rg["id"] = n
        rg["start"], rg["end"] = round(rg["start"], 2), round(rg["end"], 2)
        rg["categories"].sort(key=CATEGORIES.index)
    return [{k: rg[k] for k in ("id", "start", "end", "categories", "reasons", "segments", "hallucinated", "gaps")}
            for rg in regions]


def _stats(segs, regions, flags, duration) -> Dict[str, Any]:
    reason_counts: Dict[str, int] = {}
    for rs in flags.values():
        for r in rs:
            reason_counts[r] = reason_counts.get(r, 0) + 1
    gaps = [g for rg in regions for g in rg["gaps"]]
    if gaps:
        reason_counts["gap"] = len(gaps)
    lps = [x["avg_logprob"] for x in segs if x.get("avg_logprob") is not None]
    end = max((x["end"] for x in segs), default=0.0)
    return {
        "segments": len(segs),
        "transcript_seconds": round(duration if duration is not None else end, 2),
        "regions": len(regions),
        "regions_by_category": {c: sum(1 for rg in regions if c in rg["categories"]) for c in CATEGORIES},
        "segments_by_category": {c: sum(1 for rs in flags.values() if any(CATEGORY_OF[r] == c for r in rs))
                                 for c in CATEGORIES},
        "reason_counts": {r: reason_counts[r] for r in REASON_TEXT if r in reason_counts},
        "gap_seconds": round(sum(b - a for a, b in gaps), 2),
        "flagged_seconds": round(sum(rg["end"] - rg["start"] for rg in regions), 2),
        "mean_avg_logprob": round(sum(lps) / len(lps), 3) if lps else None,
    }
