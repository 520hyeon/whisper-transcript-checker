"""명령줄 진입점: python -m transcript_checker INPUT [옵션]"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .detect import DEFAULTS, check
from .loader import FORMATS, InputError, load
from .report import render_markdown, render_terminal

EXIT_OK, EXIT_ISSUES, EXIT_INPUT = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="whisper-check",
        description="faster-whisper 세그먼트(JSON/JSONL)에서 반복 환각, 누락 의심 구간, 인식이 불안정한 구간을 찾습니다.")
    p.add_argument("input", nargs="?", help="세그먼트 JSON 또는 JSONL 파일 (- 이면 표준 입력)")
    p.add_argument("--format", choices=FORMATS, default="auto", help="입력 형식 (기본: 내용을 보고 판단)")
    p.add_argument("-o", "--report", metavar="PATH", help="Markdown 리포트를 저장할 경로")
    p.add_argument("--json", dest="json_out", metavar="PATH", help="결과를 JSON으로 저장할 경로")
    p.add_argument("--language", help="전사 언어 코드. ko이면 영어로만 나온 세그먼트(번역 의심)도 검사 "
                                      "(기본: 입력 파일의 language)")
    p.add_argument("--duration", type=float, metavar="SEC",
                   help="녹음 전체 길이(초). 마지막 세그먼트 뒤의 간격도 검사 (기본: 입력 파일의 duration)")
    p.add_argument("--prompt", action="append", default=[], metavar="TEXT",
                   help="전사에 쓴 initial_prompt·hotwords 문구. 그대로 출력됐는지 검사 (여러 번 지정 가능)")
    p.add_argument("--prompt-file", action="append", default=[], metavar="PATH",
                   help="프롬프트 문구 파일 (한 줄에 하나)")
    p.add_argument("--phrase", action="append", default=[], metavar="TEXT", help="알려진 환각 문구 추가")
    p.add_argument("--config", metavar="PATH", help="임계값을 덮어쓸 JSON 파일 (--print-defaults로 형식 확인)")
    p.add_argument("--context", type=int, default=1, metavar="N", help="리포트에 함께 보여 줄 앞뒤 세그먼트 수 (기본 1)")
    p.add_argument("--fail-on-issues", action="store_true", help="의심 구간이 있으면 종료 코드 1")
    p.add_argument("-q", "--quiet", action="store_true", help="터미널 출력 생략")
    p.add_argument("--no-color", action="store_true", help="터미널 색상 끄기")
    p.add_argument("--print-defaults", action="store_true", help="기본 임계값을 JSON으로 출력하고 끝내기")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def load_config(path: str) -> Dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise InputError(f"설정 파일이 없습니다: {Path(path).name}") from None
    except json.JSONDecodeError as e:
        raise InputError(f"설정 파일 JSON 오류 ({e.lineno}번째 줄: {e.msg})") from None
    if not isinstance(data, dict):
        raise InputError("설정 파일은 JSON 객체여야 합니다")
    unknown = sorted(set(data) - set(DEFAULTS))
    if unknown:
        raise InputError(f"알 수 없는 설정: {', '.join(unknown)}")
    for k, v in data.items():
        if isinstance(DEFAULTS[k], list):
            if not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
                raise InputError(f"설정 '{k}'는 문자열 배열이어야 합니다")
        elif isinstance(v, bool) or not isinstance(v, (int, float)):
            raise InputError(f"설정 '{k}'는 숫자여야 합니다")
    return data


def read_lines(path: str) -> List[str]:
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise InputError(f"파일이 없습니다: {Path(path).name}") from None
    return [line.strip() for line in text.splitlines() if line.strip()]


def _use_color(stream, disabled: bool) -> bool:
    return not disabled and "NO_COLOR" not in os.environ and hasattr(stream, "isatty") and stream.isatty()


def _utf8(stream) -> None:
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _write(path: str, text: str) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    _utf8(sys.stdout)
    _utf8(sys.stderr)
    args = build_parser().parse_args(argv)
    if args.print_defaults:
        print(json.dumps(DEFAULTS, ensure_ascii=False, indent=2))
        return EXIT_OK
    if not args.input:
        build_parser().print_usage(sys.stderr)
        print("whisper-check: 오류: 입력 파일을 지정하세요", file=sys.stderr)
        return EXIT_INPUT
    if args.context < 0:
        print("whisper-check: 오류: --context는 0 이상이어야 합니다", file=sys.stderr)
        return EXIT_INPUT

    try:
        settings = load_config(args.config) if args.config else {}
        if args.phrase:
            base = settings.get("hallucination_phrases", DEFAULTS["hallucination_phrases"])
            settings["hallucination_phrases"] = list(base) + args.phrase
        prompts = list(args.prompt)
        for f in args.prompt_file:
            prompts += read_lines(f)
        transcript = load(args.input, args.format)
    except InputError as e:
        print(f"whisper-check: 오류: {e}", file=sys.stderr)
        return EXIT_INPUT

    language = args.language if args.language is not None else transcript.language
    duration = args.duration if args.duration is not None else transcript.duration
    result = check(transcript.segments, settings, language=language, prompt_texts=prompts, duration=duration)

    if not args.quiet:
        print(render_terminal(result, transcript.source, color=_use_color(sys.stdout, args.no_color)))
    if args.report:
        _write(args.report, render_markdown(result, transcript.source, context=args.context))
        if not args.quiet:
            print(f"\nMarkdown 리포트: {args.report}")
    if args.json_out:
        _write(args.json_out, json.dumps({"source": transcript.source, **result.to_dict()},
                                         ensure_ascii=False, indent=2) + "\n")
        if not args.quiet:
            print(f"JSON 결과: {args.json_out}")
    return EXIT_ISSUES if args.fail_on_issues and result.regions else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
