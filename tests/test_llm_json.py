import pytest

from lemonreel.llm_json import (
    JSONExtractionError,
    extract_json,
    extract_json_object,
    strip_reasoning,
)


def test_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_with_chatter():
    text = 'Sure! Here it is:\n```json\n{"title": "Hi", "scenes": []}\n```\nEnjoy.'
    assert extract_json(text) == {"title": "Hi", "scenes": []}


def test_reasoning_block_is_ignored():
    text = '<think>Maybe {"wrong": true}?</think>\n{"score": 4}'
    assert extract_json(text) == {"score": 4}
    assert strip_reasoning("<think>unfinished thought {") == ""


def test_braces_inside_strings():
    assert extract_json('Result: {"text": "use {braces} and \\"quotes\\""} done') == {
        "text": 'use {braces} and "quotes"'
    }


def test_repairs_common_mistakes():
    text = "{“score”: 3, “ok”: True, “items”: [1, 2,],}"
    assert extract_json(text) == {"score": 3, "ok": True, "items": [1, 2]}


def test_single_object_in_list_is_unwrapped():
    assert extract_json_object('[{"a": 1}]') == {"a": 1}


def test_failure_is_explicit():
    with pytest.raises(JSONExtractionError):
        extract_json("no json here at all")
    with pytest.raises(JSONExtractionError):
        extract_json_object("[1, 2, 3]")
