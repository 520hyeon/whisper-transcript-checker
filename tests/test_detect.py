from conftest import seg

from transcript_checker import check, load


def reasons_of(result):
    return [r for rg in result.regions for r in rg["reasons"]]


def test_clean_sample_has_no_regions(samples):
    t = load(str(samples / "clean.jsonl"))
    result = check(t.segments, language=t.language, duration=t.duration)
    assert result.regions == []
    assert result.stats["regions"] == 0


def test_problem_sample_finds_every_kind(samples):
    t = load(str(samples / "problems.json"))
    prompts = (samples / "prompt.txt").read_text(encoding="utf-8").splitlines()
    result = check(t.segments, language=t.language, prompt_texts=prompts, duration=t.duration)
    found = set(reasons_of(result))
    assert found == {"repeat", "compression", "gap", "english", "dup", "sparse", "numbers", "counting",
                     "low_logprob", "no_speech", "prompt", "phrase"}
    assert result.flags[4] == ["repeat", "compression"]
    assert result.flags[12] == ["sparse"]
    assert result.stats["regions_by_category"] == {"hallucination": 6, "omission": 3, "unstable": 2}
    # 마지막 세그먼트 뒤 14초 간격은 바로 앞 환각 문구와 한 구간으로 합쳐진다
    last = result.regions[-1]
    assert last["reasons"] == ["phrase", "gap"]
    assert last["gaps"] == [[146.0, 160.0]]
    assert last["categories"] == ["hallucination", "omission"]


def test_gaps_at_start_between_and_end():
    segs = [seg(0, 5, 8), seg(1, 8.5, 10), seg(2, 15, 18)]
    result = check(segs, duration=30)
    assert [rg["gaps"] for rg in result.regions] == [[[0.0, 5.0]], [[10.0, 15.0]], [[18.0, 30.0]]]
    assert result.stats["gap_seconds"] == 22.0


def test_no_trailing_gap_without_duration():
    result = check([seg(0, 0, 3)])
    assert result.regions == []
    assert result.stats["transcript_seconds"] == 3.0


def test_overlapping_segments_do_not_create_false_gap():
    segs = [seg(0, 0, 9, "아주 긴 문장을 여기에 적어서 글자 수를 넉넉히 채웁니다"),
            seg(1, 2, 4), seg(2, 10, 12)]  # 1은 0 안에 들어감 → 4~10초를 간격으로 보면 안 됨
    result = check(segs)
    assert result.regions == []


def test_unsorted_input_is_sorted_by_time():
    segs = [seg(0, 10, 12), seg(1, 0, 3), seg(2, 3, 10, "아주 긴 세그먼트지만 글자도 충분히 많이 들어 있습니다")]
    result = check(segs)
    assert [x["idx"] for x in result.segments] == [1, 2, 0]
    assert result.regions == []


def test_sparse_needs_long_segment_with_few_chars():
    assert reasons_of(check([seg(0, 0, 10, "음")])) == ["sparse"]
    assert check([seg(0, 0, 7, "음")]).regions == []  # 8초 미만이면 검사하지 않음
    # 환각 이유가 있으면 sparse로 따로 세지 않음
    assert reasons_of(check([seg(0, 0, 10, "시청해주셔서 감사합니다")])) == ["phrase"]


def test_nearby_items_merge_into_one_region():
    segs = [seg(0, 0, 3, "뭔가 말했다", avg_logprob=-1.5), seg(1, 3.5, 6),
            seg(2, 6.5, 9, "다시 뭔가", avg_logprob=-1.4)]
    result = check(segs)
    assert len(result.regions) == 2  # 0번 끝(3.0)과 2번 시작(6.5) 사이가 1초보다 멂
    result = check(segs, settings={"region_join_seconds": 4.0})
    assert len(result.regions) == 1
    assert result.regions[0]["segments"] == [0, 2]
    assert result.regions[0]["hallucinated"] == []


def test_settings_override_thresholds():
    segs = [seg(0, 0, 3), seg(1, 6, 9)]
    assert check(segs).regions == []
    assert reasons_of(check(segs, settings={"gap_min_seconds": 2.5})) == ["gap"]


def test_custom_phrase_setting():
    segs = [seg(0, 0, 3, "광고 문의는 아래로")]
    assert check(segs).regions == []
    phrases = {"hallucination_phrases": ["광고 문의"]}
    assert reasons_of(check(segs, settings=phrases)) == ["phrase"]


def test_empty_transcript():
    result = check([], duration=10)
    assert reasons_of(result) == ["gap"]
    assert check([]).regions == []


def test_to_dict_is_json_serialisable(samples):
    import json
    t = load(str(samples / "problems.json"))
    data = check(t.segments, language="ko", duration=t.duration).to_dict()
    assert json.loads(json.dumps(data))["stats"]["segments"] == 22
