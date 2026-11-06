import json
import subprocess
import sys

from transcript_checker.cli import EXIT_INPUT, EXIT_ISSUES, EXIT_OK, main
from transcript_checker.detect import DEFAULTS
from transcript_checker.report import fmt_time


def run(*args):
    return main([str(a) for a in args])


def test_problem_sample_writes_markdown_and_json(samples, tmp_path, capsys):
    md, js = tmp_path / "out" / "report.md", tmp_path / "result.json"
    code = run(samples / "problems.json", "--prompt-file", samples / "prompt.txt", "-o", md, "--json", js, "--no-color")
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "의심 구간 9개" in out and "\033[" not in out

    report = md.read_text(encoding="utf-8")
    assert report.startswith("# Whisper 전사 점검 리포트")
    for heading in ("## 요약", "## 의심 구간", "## 구간별 상세", "## 사용한 기준"):
        assert heading in report
    assert "- 입력: `problems.json`" in report
    assert str(samples) not in report and str(tmp_path) not in report  # 개인 경로가 남지 않음
    assert "(전사 없음 14.0초)" in report
    assert "`prompt`" in report

    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["source"] == "problems.json"
    assert data["stats"]["regions"] == 9
    assert {"idx": 19, "reasons": ["prompt"]} in data["flagged_segments"]


def test_fail_on_issues_exit_codes(samples, capsys):
    assert run(samples / "problems.json", "--fail-on-issues", "-q") == EXIT_ISSUES
    assert run(samples / "clean.jsonl", "--fail-on-issues", "-q") == EXIT_OK
    assert capsys.readouterr().out == ""


def test_clean_report(samples, tmp_path):
    md = tmp_path / "clean.md"
    assert run(samples / "clean.jsonl", "-o", md, "-q") == EXIT_OK
    assert "의심 구간이 없습니다." in md.read_text(encoding="utf-8")


def test_cli_options_override_file_metadata(samples, capsys):
    # language를 en으로 바꾸면 영어 세그먼트를 번역 의심으로 보지 않음
    run(samples / "problems.json", "--language", "en", "--no-color")
    assert "영어로만 출력" not in capsys.readouterr().out
    # duration을 마지막 세그먼트 끝으로 주면 끝 간격이 사라짐
    run(samples / "problems.json", "--duration", "146", "--no-color")
    assert "(전사 없음 14.0초)" not in capsys.readouterr().out


def test_prompt_and_phrase_options(samples, capsys):
    run(samples / "problems.json", "--no-color")
    assert "프롬프트 누출" not in capsys.readouterr().out
    run(samples / "problems.json", "--no-color", "--prompt", "상자 텃밭 배수층 순지르기 웃거름")
    assert "프롬프트 누출" in capsys.readouterr().out
    run(samples / "clean.jsonl", "--no-color", "--phrase", "알려 드릴게요")
    assert "알려진 환각 문구" in capsys.readouterr().out


def test_config_file(samples, tmp_path, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"gap_min_seconds": 1.0}), encoding="utf-8")
    run(samples / "clean.jsonl", "--config", cfg, "--no-color")
    assert "전사 없는 간격" in capsys.readouterr().out


def test_bad_config_is_input_error(samples, tmp_path, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"gap_seconds": 1.0}), encoding="utf-8")
    assert run(samples / "clean.jsonl", "--config", cfg) == EXIT_INPUT
    assert "알 수 없는 설정: gap_seconds" in capsys.readouterr().err
    cfg.write_text(json.dumps({"gap_min_seconds": "1"}), encoding="utf-8")
    assert run(samples / "clean.jsonl", "--config", cfg) == EXIT_INPUT


def test_bad_input_is_exit_2(tmp_path, capsys):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"start": 0, "end": 1}\nnot json\n', encoding="utf-8")
    assert run(p) == EXIT_INPUT
    assert "2번째 줄" in capsys.readouterr().err
    assert run() == EXIT_INPUT


def test_print_defaults(capsys):
    assert run("--print-defaults") == EXIT_OK
    assert json.loads(capsys.readouterr().out) == DEFAULTS


def test_module_entry_point_reads_stdin(samples):
    data = (samples / "clean.jsonl").read_bytes()
    proc = subprocess.run([sys.executable, "-m", "transcript_checker", "-", "--no-color"],
                          input=data, capture_output=True, cwd=samples.parent)
    assert proc.returncode == 0
    assert "의심 구간이 없습니다." in proc.stdout.decode("utf-8")


def test_fmt_time():
    assert fmt_time(0) == "00:00.0"
    assert fmt_time(75.3) == "01:15.3"
    assert fmt_time(59.96) == "01:00.0"  # 반올림으로 분이 넘어가도 60.0초로 표시하지 않음
    assert fmt_time(3725.0) == "1:02:05.0"
