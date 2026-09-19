"""Tests for argus.llm.output_models: schema generation and Responses API format.

Verifies pydantic_to_response_format (including exclude= and strict= handling)
and LLMOutputModel schema generation for compatibility with OpenAI Responses API.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

from argus.graph import _PIPELINE_ONLY_RESPONSE_FIELDS
from argus.llm.output_models import (
    LLMOutputModel,
    make_schema_strict,
    pydantic_to_response_format,
)
from argus.models import ReviewResponse


class SampleModel(BaseModel):
    name: str = Field(description="The name")
    count: int = Field(default=0, description="The count")
    tag: str | None = None


class SampleModelWithExcludedFields(BaseModel):
    title: str
    verdict: str
    dropped_field_1: dict[str, float] = Field(default_factory=dict)
    dropped_field_2: dict[str, float] = Field(default_factory=dict)


class SampleOutputModel(LLMOutputModel):
    _schema_name = "custom_output"

    summary: str = Field(description="Summary of results")
    status: Literal["pass", "fail"] = Field(description="Pass or fail")


class TestPydanticToResponseFormat:
    def test_default_name_and_strict(self) -> None:
        """pydantic_to_response_format defaults name to model name in lowercase and strict=True."""
        rf = pydantic_to_response_format(SampleModel)

        assert rf["type"] == "json_schema"
        assert rf["name"] == "samplemodel"
        assert rf["strict"] is True
        schema = rf["schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"].keys())

    def test_custom_name(self) -> None:
        """pydantic_to_response_format respects custom name."""
        rf = pydantic_to_response_format(SampleModel, name="my_custom_schema")
        assert rf["name"] == "my_custom_schema"

    def test_strict_false(self) -> None:
        """pydantic_to_response_format respects strict=False."""
        rf = pydantic_to_response_format(SampleModel, strict=False)
        assert rf["strict"] is False
        # strict=False does not enforce additionalProperties: false
        assert rf["schema"].get("additionalProperties") is not False

    def test_exclude_fields(self) -> None:
        """exclude= removes specified fields from schema properties and required."""
        rf = pydantic_to_response_format(
            SampleModelWithExcludedFields,
            name="filtered_model",
            exclude={"dropped_field_1", "dropped_field_2"},
        )

        properties = rf["schema"]["properties"]
        assert "title" in properties
        assert "verdict" in properties
        assert "dropped_field_1" not in properties
        assert "dropped_field_2" not in properties

        required = rf["schema"]["required"]
        assert "title" in required
        assert "verdict" in required
        assert "dropped_field_1" not in required
        assert "dropped_field_2" not in required

    def test_review_response_with_pipeline_only_fields_excluded(self) -> None:
        """ReviewResponse with _PIPELINE_ONLY_RESPONSE_FIELDS matches production graph.py usage."""
        rf = pydantic_to_response_format(
            ReviewResponse,
            "review_response",
            exclude=_PIPELINE_ONLY_RESPONSE_FIELDS,
        )

        assert rf["type"] == "json_schema"
        assert rf["name"] == "review_response"
        assert rf["strict"] is True

        properties = rf["schema"]["properties"]
        required = rf["schema"]["required"]

        for excluded in _PIPELINE_ONLY_RESPONSE_FIELDS:
            assert excluded not in properties
            assert excluded not in required

        # Essential ReviewResponse fields must remain present and required
        assert "verdict" in properties
        assert "verdict" in required
        assert "risk_level" in properties
        assert "risk_level" in required
        assert "findings" in properties
        assert "findings" in required


class TestLLMOutputModel:
    def test_to_prompt_schema(self) -> None:
        """to_prompt_schema generates markdown with field definitions and example."""
        schema_text = SampleOutputModel.to_prompt_schema()
        assert "Your output MUST be valid JSON" in schema_text
        assert "| `summary` | string |" in schema_text
        assert "| `status` | enum |" in schema_text
        assert (
            "custom_output" not in schema_text
        )  # schema name is in API response_format, not prompt

    def test_to_response_format(self) -> None:
        """to_response_format produces Chat Completions nested format."""
        rf = SampleOutputModel.to_response_format()
        assert rf["type"] == "json_schema"
        assert "json_schema" in rf
        assert rf["json_schema"]["name"] == "custom_output"
        assert rf["json_schema"]["strict"] is True

    def test_to_response_format_flat(self) -> None:
        """to_response_format_flat produces Responses API flat format."""
        rf = SampleOutputModel.to_response_format_flat()
        assert rf["type"] == "json_schema"
        assert rf["name"] == "custom_output"
        assert rf["strict"] is True
        assert "schema" in rf
        assert rf["schema"]["type"] == "object"
        assert rf["schema"]["additionalProperties"] is False


class TestMakeSchemaStrict:
    def test_recursively_sets_additional_properties_false_and_required(self) -> None:
        """make_schema_strict sets additionalProperties=False and required on all objects."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                        },
                    },
                },
            },
        }

        strict_schema = make_schema_strict(schema)

        assert strict_schema["additionalProperties"] is False
        assert set(strict_schema["required"]) == {"user", "items"}

        user_prop = strict_schema["properties"]["user"]
        assert user_prop["additionalProperties"] is False
        assert user_prop["required"] == ["name"]

        item_schema = strict_schema["properties"]["items"]["items"]
        assert item_schema["additionalProperties"] is False
        assert item_schema["required"] == ["id"]

    def test_ref_clears_sibling_keywords(self) -> None:
        """OpenAI strict mode requires $ref to have no sibling keywords."""
        schema = {
            "$ref": "#/$defs/SomeType",
            "description": "Sibling keyword that OpenAI rejects in strict mode",
        }
        strict_schema = make_schema_strict(schema)
        assert strict_schema == {"$ref": "#/$defs/SomeType"}
