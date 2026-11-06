"""검사 결과를 터미널 텍스트와 Markdown 리포트로 만든다."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .detect import CATEGORIES, CATEGORY_OF, CATEGORY_TEXT, REASON_TEXT, CheckResult

_COLOR = {"hallucination": "31", "omission": "33", "unstable": "36", "dim": "2", "bold": "1"}
_SHORT = {"hallucination": "환각", "omission": "누락", "unstable": "불안정"}


def fmt_time(sec: float) -> str:
    """초 → mm:ss.s (1시간 이상이면 h:mm:ss.s)."""
    tenths = int(round(max(sec, 0.0) * 10))
    h, rest = divmod(tenths, 36000)
    m, rest = divmod(rest, 600)
    s = rest / 10
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m:02d}:{s:04.1f}"


def fmt_span(a: float, b: float) -> str:
    return f"{fmt_time(a)}–{fmt_time(b)}"


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def _metric(v: Optional[float], digits: int = 2) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def _reason_list(reasons: List[str]) -> str:
    return ", ".join(REASON_TEXT[r] for r in reasons)


def _category_summary(counts: Dict[str, int]) -> str:
    return " · ".join(f"{_SHORT[c]} {counts[c]}" for c in CATEGORIES if counts.get(c))


# ---------- 터미널 ----------

def render_terminal(result: CheckResult, source: str = "-", color: bool = False, max_text: int = 48) -> str:
    def c(code, text):
        return f"\033[{_COLOR[code]}m{text}\033[0m" if color else text

    st = result.stats
    by_idx = {x["idx"]: x for x in result.segments}
    lines = [c("bold", f"whisper-transcript-checker · {source}"),
             f"세그먼트 {st['segments']}개 · 길이 {fmt_time(st['transcript_seconds'])}"
             + (f" · 언어 {result.language}" if result.language else "")
             + (f" · 평균 avg_logprob {st['mean_avg_logprob']:.3f}" if st["mean_avg_logprob"] is not None else ""),
             ""]
    if not result.regions:
        lines.append("의심 구간이 없습니다.")
        return "\n".join(lines)
    lines.append(f"의심 구간 {st['regions']}개 ({_category_summary(st['regions_by_category'])}) · "
                 f"합계 {st['flagged_seconds']:.1f}초")
    for rg in result.regions:
        cats = "/".join(c(k, _SHORT[k]) for k in rg["categories"])
        lines.append(f"  [{rg['id']:>2}] {fmt_span(rg['start'], rg['end'])}  {cats}  {_reason_list(rg['reasons'])}")
        for i in rg["segments"]:
            seg = by_idx[i]
            lines.append(c("dim", f"       #{i} {_clip(seg['text'], max_text)!r}"))
        for a, b in rg["gaps"]:
            lines.append(c("dim", f"       (전사 없음 {b - a:.1f}초)"))
    return "\n".join(lines)


# ---------- Markdown ----------

def _cell(text: Any) -> str:
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("<", "&lt;")


def _table(header: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_cell(x) for x in row) + " |" for row in rows]
    return out


def render_markdown(result: CheckResult, source: str = "-", context: int = 1, max_text: int = 120) -> str:
    st = result.stats
    segs = result.segments
    pos = {x["idx"]: k for k, x in enumerate(segs)}
    out = ["# Whisper 전사 점검 리포트", ""]
    meta = [f"- 입력: `{source}`", f"- 세그먼트: {st['segments']}개",
            f"- 길이: {fmt_time(st['transcript_seconds'])}"
            + (" (duration 기준)" if result.duration is not None else " (마지막 세그먼트 끝 기준)")]
    if result.language:
        meta.append(f"- 언어: `{result.language}`")
    if st["mean_avg_logprob"] is not None:
        meta.append(f"- 평균 avg_logprob: {st['mean_avg_logprob']:.3f}")
    if result.prompt_texts:
        meta.append(f"- 프롬프트 누출 검사: 문구 {len(result.prompt_texts)}개")
    out += meta + ["", "## 요약", ""]

    rows = [[CATEGORY_TEXT[c], st["regions_by_category"][c], st["segments_by_category"][c]] for c in CATEGORIES]
    rows.append(["**전체**", f"**{st['regions']}**", f"**{len(result.flags)}**"])
    out += _table(["분류", "구간", "세그먼트"], rows)
    out += ["", f"의심 구간 합계 {st['flagged_seconds']:.1f}초"
            + (f", 그중 전사 없는 간격 {st['gap_seconds']:.1f}초" if st["gap_seconds"] else "") + "."]
    if st["reason_counts"]:
        out += ["", "| 이유 | 코드 | 분류 | 횟수 |", "|---|---|---|---|"]
        out += [f"| {REASON_TEXT[r]} | `{r}` | {_SHORT[CATEGORY_OF[r]]} | {n} |" for r, n in st["reason_counts"].items()]

    if not result.regions:
        out += ["", "의심 구간이 없습니다.", ""]
        return "\n".join(out)

    out += ["", "## 의심 구간", ""]
    rows = [[rg["id"], fmt_span(rg["start"], rg["end"]), f"{rg['end'] - rg['start']:.1f}초",
             ", ".join(_SHORT[c] for c in rg["categories"]), _reason_list(rg["reasons"]),
             ", ".join(f"#{i}" for i in rg["segments"]) or "—"] for rg in result.regions]
    out += _table(["#", "시간", "길이", "분류", "이유", "세그먼트"], rows)

    out += ["", "## 구간별 상세", ""]
    if context:
        out += [f"_앞뒤 문맥 세그먼트 {context}개씩 함께 표시합니다 (이유 칸이 비어 있음)._", ""]
    for rg in result.regions:
        out += [f"### {rg['id']}. {fmt_span(rg['start'], rg['end'])} · "
                + ", ".join(CATEGORY_TEXT[c] for c in rg["categories"]), ""]
        out += [f"- {REASON_TEXT[r]} (`{r}`)" for r in rg["reasons"]] + [""]
        in_region = set(rg["segments"])
        ks = sorted(pos[i] for i in in_region)
        if ks:
            lo, hi = max(0, ks[0] - context), min(len(segs) - 1, ks[-1] + context)
        else:  # 간격만 있는 구간: 간격 앞뒤 세그먼트를 문맥으로
            after = [k for k, x in enumerate(segs) if x["start"] >= rg["start"]]
            first_after = after[0] if after else len(segs)
            lo, hi = max(0, first_after - context), min(len(segs) - 1, first_after + context - 1)
        entries = []  # (start, 순서, row)
        for k in range(lo, hi + 1):
            x = segs[k]
            flagged = x["idx"] in in_region
            entries.append((x["start"], 1, [
                f"**#{x['idx']}**" if flagged else f"#{x['idx']}", fmt_span(x["start"], x["end"]),
                _metric(x.get("avg_logprob"), 3), _metric(x.get("no_speech_prob")),
                _metric(x.get("compression_ratio")),
                ", ".join(f"`{r}`" for r in result.flags.get(x["idx"], [])) if flagged else "",
                _clip(x["text"], max_text) or "(빈 텍스트)"]))
        for a, b in rg["gaps"]:
            entries.append((a, 0, ["—", fmt_span(a, b), "", "", "", "`gap`", f"(전사 없음 {b - a:.1f}초)"]))
        entries.sort(key=lambda e: (e[0], e[1]))
        out += _table(["#", "시간", "avg_logprob", "no_speech", "압축률", "이유", "텍스트"], [e[2] for e in entries])
        out.append("")

    out += ["## 사용한 기준", "", "<details><summary>임계값 보기</summary>", "", "```json",
            json.dumps(result.settings, ensure_ascii=False, indent=2), "```", "", "</details>", ""]
    return "\n".join(out)
