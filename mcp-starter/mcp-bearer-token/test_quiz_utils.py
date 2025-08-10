import pytest
import os
import sys

sys.path.append(os.path.dirname(__file__))
import quiz_utils as q


def test_normalize_axis():
    assert q.normalize_axis("ei") == "EI"
    assert q.normalize_axis(" Sn ") == "SN"
    assert q.normalize_axis("xx") is None


@pytest.mark.parametrize(
    "inp,expected",
    [
        (3, 3.0),
        (-4, -3.0),
        ("a", -3.0),
        ("G", 3.0),
        ("strongly agree", 3.0),
        ("Slightly Agree", 1.0),
        ("Neutral", 0.0),
        ("slightly   disagree", -1.0),
        ("Strong Disagree", -3.0),
        ("disagree", -2.0),
        ("ei: a", -3.0),
        (" value = 2 ", 2.0),
        ("junk 1.5 text", 1.5),
        (None, None),
    ],
)
def test_sanitize_value(inp, expected):
    assert q.sanitize_value(inp) == expected


def test_sanitize_responses_mixed():
    inputs = [
        {"axis": "ei", "value": "A"},
        {"AXIS": "SN", "VALUE": -1},
        {"id": "TF-2", "answer": "c"},
        "jp = strongly agree",
        "bad = noop",
    ]
    out = q.sanitize_responses(inputs)
    # Expect four valid entries
    assert len(out) == 4
    axes = {x["axis"] for x in out}
    assert axes == {"EI", "SN", "TF", "JP"}


def test_parse_compact_tokens():
    text = "1a  2:b  3-c  4 = D  5 -2  6 neutral 7 strongly agree  8 1.5  xx 6Z"
    parsed = q.parse_compact_tokens(text)
    # Order by first match per index, no duplicates
    assert (1, "a") in parsed
    assert (2, "b") in parsed
    assert (3, "c") in parsed
    assert (4, "D") in parsed
    assert (5, "-2") in parsed
    # For textual forms we accept the raw phrase; sanitizer will convert later
    assert any(idx == 6 and isinstance(val, str) and "neutral" in val.lower() for idx, val in parsed)
    assert any(idx == 7 and isinstance(val, str) and "agree" in val.lower() for idx, val in parsed)


