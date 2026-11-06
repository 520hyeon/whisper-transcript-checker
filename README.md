# Whisper Transcript Checker

faster-whisper가 만든 세그먼트(JSON 또는 JSONL)를 읽고, 사람이 다시 들어 봐야 할 구간을 찾아 주는 명령줄 도구입니다. 오디오 파일이나 GPU 없이 세그먼트 텍스트와 Whisper가 함께 내보내는 지표만으로 동작하며, 표준 라이브러리 외 의존성이 없습니다.

`lecture_transcriber`의 환각·누락 감지 로직을 전사 파이프라인과 분리해, 어떤 faster-whisper 결과에도 쓸 수 있게 만든 것입니다.

## 찾는 것

| 분류 | 코드 | 기준 (기본값) |
|---|---|---|
| **환각 의심** | `repeat` | 같은 단어·구절이 연속 3번 이상 반복되고, 반복 부분이 4단어 이상이면서 문장의 절반 이상 |
| | `dup` | 바로 앞이나 뒤 세그먼트와 같은 문장 (3단어 또는 8글자 이상) |
| | `prompt` | `--prompt`로 준 initial_prompt·hotwords 문구가 3단어 이상 이어서 나오고 문장의 60% 이상 |
| | `english` | 언어가 `ko`인데 한글 없이 영어 단어만 4개 이상 (한국어 발화를 번역해 출력한 경우) |
| | `compression` | `compression_ratio` > 2.4 |
| | `phrase` | "시청해주셔서 감사합니다" 같은 알려진 환각 문구 (띄어쓰기 무시) |
| | `numbers` | 숫자가 12개 이상이고 토큰의 80% 이상 |
| | `counting` | 1씩 늘어나는 숫자가 10개 이상 연속 |
| | `same_token` | 토큰 16개 이상인 세그먼트에서 같은 형태(숫자는 `#`)의 토큰이 절반 이상 |
| **누락 의심** | `gap` | 세그먼트 사이(그리고 녹음 처음·끝)에 전사 없는 간격이 4초 이상 |
| | `sparse` | 8초 이상인 세그먼트의 글자 수가 초당 1.2자 미만 (환각 이유가 없는 경우만) |
| **인식 불안정** | `low_logprob` | `avg_logprob` < -1.0 |
| | `no_speech` | `no_speech_prob` ≥ 0.6인데 텍스트가 있음 |

1초 이내로 붙어 있는 의심 항목은 하나의 **구간**으로 합쳐 보여 줍니다. 모든 기준은 설정 파일로 바꿀 수 있습니다.

## 설치

Python 3.9 이상이면 설치 없이 이 폴더에서 바로 실행할 수 있습니다.

```bash
python -m transcript_checker samples/problems.json
```

`whisper-check` 명령으로 쓰려면 설치합니다.

```bash
pip install .            # 실행만
pip install -e ".[dev]"  # 개발 (pytest 포함)
```

## 사용법

```bash
# 터미널에 요약 출력
whisper-check segments.jsonl

# Markdown 리포트와 JSON 결과 저장
whisper-check segments.jsonl -o report.md --json result.json

# 전사할 때 쓴 프롬프트가 그대로 새어 나왔는지도 검사
whisper-check segments.json --prompt "용어1, 용어2, 용어3" --prompt-file prompt.txt

# 의심 구간이 있으면 종료 코드 1 (스크립트·CI용)
whisper-check segments.jsonl --fail-on-issues -q

# 표준 입력
cat segments.jsonl | whisper-check -
```

| 옵션 | 설명 |
|---|---|
| `-o, --report PATH` | Markdown 리포트 저장 |
| `--json PATH` | 구간·통계·세그먼트별 이유를 JSON으로 저장 |
| `--format {auto,json,jsonl}` | 입력 형식 (기본은 내용을 보고 판단) |
| `--language CODE` | 전사 언어. `ko`일 때만 `english` 검사를 함 (기본: 입력 파일의 `language`) |
| `--duration SEC` | 녹음 전체 길이. 마지막 세그먼트 뒤의 간격도 검사 (기본: 입력 파일의 `duration`) |
| `--prompt TEXT`, `--prompt-file PATH` | 프롬프트 누출 검사에 쓸 문구 (여러 번 지정 가능, 파일은 한 줄에 하나) |
| `--phrase TEXT` | 알려진 환각 문구 추가 |
| `--config PATH` | 임계값을 덮어쓸 JSON 파일 |
| `--context N` | 리포트에 함께 보여 줄 앞뒤 문맥 세그먼트 수 (기본 1) |
| `--fail-on-issues` | 의심 구간이 있으면 종료 코드 1 |
| `-q, --quiet` / `--no-color` | 터미널 출력 생략 / 색상 끄기 (`NO_COLOR` 환경 변수도 인식) |
| `--print-defaults` | 기본 임계값을 JSON으로 출력 |

종료 코드는 `0` 정상, `1` 의심 구간 있음(`--fail-on-issues`일 때만), `2` 입력·설정 오류입니다.

## 입력 형식

세그먼트마다 `start`, `end`(초)가 필요하고, `text`와 지표(`avg_logprob`, `no_speech_prob`, `compression_ratio`)는 있으면 사용합니다. 지표가 없는 세그먼트는 해당 검사만 건너뜁니다. `words`, `tokens` 같은 다른 필드는 무시합니다.

**JSON** — 세그먼트 배열, 또는 `segments` 배열을 가진 객체. `language`·`duration`은 최상위나 `info` 안에 둘 수 있습니다.

```json
{
  "language": "ko",
  "duration": 160.0,
  "segments": [
    {"start": 0.0, "end": 4.2, "text": "안녕하세요.", "avg_logprob": -0.21, "no_speech_prob": 0.02, "compression_ratio": 1.32}
  ]
}
```

**JSONL** — 한 줄에 세그먼트 하나. `start` 없이 `language`나 `duration`만 있는 줄은 메타데이터로 읽습니다.

```
{"language": "ko", "duration": 62.0}
{"start": 0.4, "end": 5.1, "text": "...", "avg_logprob": -0.19, "no_speech_prob": 0.02, "compression_ratio": 1.27}
```

리포트의 `#n`은 **파일 안 세그먼트 순서(0부터)**입니다. 입력의 `id` 필드와는 다를 수 있습니다.

faster-whisper 결과를 JSONL로 저장하는 예:

```python
import json
from faster_whisper import WhisperModel

model = WhisperModel("large-v3")
segments, info = model.transcribe("audio.wav", language="ko")
with open("segments.jsonl", "w", encoding="utf-8") as f:
    f.write(json.dumps({"language": info.language, "duration": info.duration}) + "\n")
    for s in segments:
        f.write(json.dumps({"start": s.start, "end": s.end, "text": s.text,
                            "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                            "compression_ratio": s.compression_ratio}, ensure_ascii=False) + "\n")
```

## 출력

터미널 출력 (`samples/problems.json`):

```
whisper-transcript-checker · problems.json
세그먼트 22개 · 길이 02:40.0 · 언어 ko · 평균 avg_logprob -0.428

의심 구간 9개 (환각 6 · 누락 3 · 불안정 2) · 합계 91.6초
  [ 1] 00:21.0–00:29.5  환각  같은 말 반복, 압축률 과다
       #4 '물을 너무 자주 주면 물을 너무 자주 주면 물을 너무 자주 주면 물을 너무 자주 주면'
  [ 2] 00:35.0–00:44.0  누락  전사 없는 간격
       (전사 없음 9.0초)
  ...
  [ 9] 02:22.0–02:40.0  환각/누락  알려진 환각 문구, 전사 없는 간격
       #21 '시청해주셔서 감사합니다.'
       (전사 없음 14.0초)
```

Markdown 리포트에는 요약표, 이유별 횟수, 구간 목록, 구간마다 앞뒤 문맥 세그먼트와 지표를 담은 상세표, 사용한 임계값이 들어갑니다. 전체 예시는 [samples/report_example.md](samples/report_example.md)에 있습니다. 리포트에는 입력 파일의 경로 대신 파일 이름만 적습니다.

## 설정

`--print-defaults`로 기본값을 확인하고, 바꿀 항목만 담은 JSON을 `--config`로 넘깁니다. 모르는 키나 잘못된 타입은 오류로 처리합니다.

```json
{
  "gap_min_seconds": 6.0,
  "low_logprob": -0.8,
  "hallucination_phrases": ["시청해주셔서 감사합니다", "다음 영상에서 만나요"]
}
```

`hallucination_phrases`를 설정 파일에 넣으면 기본 목록을 대체하고, `--phrase`는 목록에 추가합니다.

## 한계

- 오디오를 보지 않으므로 `gap`·`sparse`는 실제로 말이 없었던 구간(쉬는 시간, 판서, 녹음 시작 전 대기)도 잡습니다. 원래 도구는 소리 크기로 이런 구간을 걸렀지만 여기서는 후보로만 보여 줍니다. 녹음 특성에 맞게 `gap_min_seconds`를 조절하세요.
- `english` 검사는 한국어 전사(`ko`)에만 적용됩니다. 영어 슬라이드를 그대로 읽은 실제 영어 발화도 후보로 잡힐 수 있습니다.
- 감지만 하고 고치지는 않습니다. 재전사로 구간을 복원하는 기능은 포함하지 않았습니다.

## 구조

```
transcript_checker/
  detect.py    감지 규칙, 기본 임계값, 구간 합치기, 통계
  loader.py    JSON·JSONL 읽기와 검증
  report.py    터미널·Markdown 출력
  cli.py       명령줄 옵션
samples/
  problems.json       모든 감지 유형이 들어 있는 예시 (직접 만든 가상 데이터)
  clean.jsonl         의심 구간이 없어야 하는 예시
  prompt.txt          problems.json 검사용 프롬프트 문구
  report_example.md   problems.json으로 만든 리포트
tests/                pytest 테스트
```

샘플은 모두 테스트용으로 직접 작성한 가상의 전사 데이터입니다.

## 테스트

```bash
pip install -e ".[dev]"
pytest
```
