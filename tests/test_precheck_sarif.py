"""Unit tests for argus.precheck.sarif.parse_semgrep_sarif."""

from __future__ import annotations

import json

from argus.precheck.sarif import SarifResult, parse_semgrep_sarif


def _sarif_doc(rule_id: str = "no-hardcoded-secret", level: str = "error") -> str:
    return json.dumps(
        {
            "version": "2.1.0",
            "runs": [
                {
                    "results": [
                        {
                            "ruleId": rule_id,
                            "level": level,
                            "message": {"text": "hardcoded secret found"},
                            "locations": [
                                {
                                    "physicalLocation": {
                                        "artifactLocation": {"uri": "app/config.py"},
                                        "region": {"startLine": 42},
                                    }
                                }
                            ],
                        }
                    ]
                }
            ],
        }
    )


def test_parses_a_single_result() -> None:
    results = parse_semgrep_sarif(_sarif_doc())
    assert results == [
        SarifResult(
            rule_id="no-hardcoded-secret",
            level="error",
            message="hardcoded secret found",
            file="app/config.py",
            line=42,
        )
    ]


def test_empty_input_returns_empty_list() -> None:
    assert parse_semgrep_sarif("") == []
    assert parse_semgrep_sarif(b"") == []


def test_malformed_json_returns_empty_list_not_raise() -> None:
    assert parse_semgrep_sarif("{not valid json") == []


def test_missing_rule_id_is_skipped() -> None:
    doc = json.dumps({"runs": [{"results": [{"message": {"text": "no id"}}]}]})
    assert parse_semgrep_sarif(doc) == []


def test_missing_locations_yields_none_file_and_line() -> None:
    doc = json.dumps(
        {"runs": [{"results": [{"ruleId": "r1", "message": {"text": "m"}, "level": "warning"}]}]}
    )
    results = parse_semgrep_sarif(doc)
    assert results == [
        SarifResult(rule_id="r1", level="warning", message="m", file=None, line=None)
    ]


def test_multiple_runs_and_results_all_collected() -> None:
    doc = json.dumps(
        {
            "runs": [
                {"results": [{"ruleId": "r1", "message": {"text": "a"}, "level": "error"}]},
                {"results": [{"ruleId": "r2", "message": {"text": "b"}, "level": "note"}]},
            ]
        }
    )
    results = parse_semgrep_sarif(doc)
    assert [r.rule_id for r in results] == ["r1", "r2"]


def test_as_finding_dict_shape() -> None:
    result = SarifResult(rule_id="r1", level="error", message="bad thing", file="a.py", line=5)
    assert result.as_finding_dict() == {
        "file": "a.py",
        "line": 5,
        "description": "bad thing",
        "context": "deterministic precheck rule: r1",
    }


def test_as_storage_dict_shape() -> None:
    result = SarifResult(rule_id="r1", level="error", message="bad thing", file="a.py", line=5)
    assert result.as_storage_dict() == {
        "rule_id": "r1",
        "level": "error",
        "message": "bad thing",
        "file": "a.py",
        "line": 5,
    }
