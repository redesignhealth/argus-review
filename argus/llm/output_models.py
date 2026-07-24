"""Base classes for LLM structured output models.

This module provides a standardized pattern for defining Pydantic models used with
OpenAI's structured output feature. It implements "dual enforcement":

1. **Prompt Guidance**: Human-readable schema in the prompt helps the LLM understand
   the expected output structure, constraints, and examples.

2. **API Enforcement**: JSON schema passed to OpenAI's `response_format` parameter
   enforces the structure at the API level.

Usage:
    from argus.llm.output_models import LLMOutputModel
    from pydantic import Field
    from typing import Literal

    class MyAgentOutput(LLMOutputModel):
        '''Output from my agent.'''

        summary: str = Field(description="2-3 paragraph summary of findings")
        key_points: list[str] = Field(
            min_length=3,
            max_length=5,
            description="3-5 key points with citations"
        )
        recommendation: Literal["Approve", "Reject", "Review"] = Field(
            description="Final recommendation"
        )
        confidence: Literal["High", "Medium", "Low"] = Field(
            description="Confidence level in the assessment"
        )

    # In your flow/service:
    # 1. Generate prompt schema
    schema_appendix = MyAgentOutput.to_prompt_schema()
    full_prompt = f"{task_prompt}\\n\\n{schema_appendix}\\n\\nInput: {input_data}"

    # 2. Get response format for OpenAI (strict=True by default)
    response_format = MyAgentOutput.to_response_format()

    # 3. Call OpenAI and parse
    response = await openai_call(prompt=full_prompt, response_format=response_format)
    result = MyAgentOutput.model_validate_json(response)
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, get_args, get_origin

from pydantic import BaseModel


class LLMOutputModel(BaseModel):
    """Base class for LLM structured output models.

    Inherit from this class to get automatic schema generation for prompts
    and OpenAI structured output enforcement.

    Features:
    - `to_prompt_schema()`: Generates human-readable schema for prompt inclusion
    - `to_response_format()`: Returns OpenAI response_format dict for Chat Completions API
    - `to_response_format_flat()`: Returns response_format dict for Responses API
    - Automatic field documentation from Field(description=...)
    - Support for Literal types, lists, nested models

    Schema Naming Convention:
        Override `_schema_name` to set a custom schema name used in `response_format`.
        Use snake_case for consistency (e.g., "meeting_filter_result", "agent_commentary").
        If not set, defaults to the class name in lowercase (e.g., "MyModel" -> "mymodel").

        Examples:
            _schema_name: ClassVar[str] = "concept_extraction"
            _schema_name: ClassVar[str] = "committee_organizer_output"
    """

    # Override in subclasses to customize the schema name used in response_format.
    # Convention: use snake_case (e.g., "meeting_filter_result", "agent_commentary").
    # Default: class name in lowercase (e.g., "MyModel" -> "mymodel").
    _schema_name: ClassVar[str] = ""

    @classmethod
    def to_prompt_schema(
        cls,
        *,
        include_example: bool = True,
        example_overrides: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> str:
        """Generate human-readable schema description for prompt inclusion.

        This method introspects the Pydantic model's fields and generates a
        comprehensive schema description that helps the LLM understand the
        expected output structure.

        Subclasses can override this method with additional parameters
        (e.g., agent_type for agent-specific examples).

        Args:
            include_example: Whether to include an example JSON output
            example_overrides: Dict of field_name -> example_value to override defaults
            **kwargs: Additional arguments for subclass customization

        Returns:
            Human-readable schema with field definitions and optional example
        """
        lines = [
            "Your output MUST be valid JSON matching the following schema exactly. "
            "This schema is strictly enforced.",
            "",
            "### Field Definitions",
            "",
            "| Field | Type | Constraints | Description |",
            "|-------|------|-------------|-------------|",
        ]

        example_data: dict[str, Any] = {}
        overrides = example_overrides or {}

        for field_name, field_info in cls.model_fields.items():
            # Get type string
            type_str = _get_type_string(field_info.annotation)

            # Get constraints
            constraints = _get_constraints(field_info)

            # Get description
            description = field_info.description or ""

            lines.append(f"| `{field_name}` | {type_str} | {constraints} | {description} |")

            # Build example value
            if field_name in overrides:
                example_data[field_name] = overrides[field_name]
            else:
                example_data[field_name] = _get_example_value(field_info)

        # Add Literal value documentation
        literal_docs = _get_literal_documentation(cls)
        if literal_docs:
            lines.append("")
            lines.extend(literal_docs)

        # Add example
        if include_example:
            lines.append("")
            lines.append("### Example Output")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(example_data, indent=4, default=str))
            lines.append("```")

        lines.append("")
        lines.append(
            "**IMPORTANT**: Your response must be ONLY the JSON object above. "
            "Do not include any text before or after the JSON."
        )

        return "\n".join(lines)

    @classmethod
    def to_response_format(cls, *, strict: bool = True) -> dict[str, Any]:
        """Generate OpenAI response_format dict for structured output.

        Args:
            strict: If True (default), uses OpenAI strict mode which:
                   - Sets additionalProperties: false on all objects
                   - Ensures all properties are in 'required' array
                   Set to False only if model has dynamic keys (e.g., dict[str, float]).

        Returns:
            Dict suitable for OpenAI's response_format parameter
        """
        schema_name = cls._schema_name or cls.__name__.lower()
        schema = cls.model_json_schema()

        if strict:
            schema = make_schema_strict(schema)

        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": strict,
            },
        }

    @classmethod
    def to_response_format_flat(cls, *, strict: bool = True) -> dict[str, Any]:
        """Generate flat response_format for OpenAI Responses API.

        The Responses API expects a flat structure with name/strict/schema at the
        top level, NOT nested under "json_schema" like the Chat Completions API.

        Args:
            strict: If True (default), uses OpenAI strict mode.

        Returns:
            Dict suitable for OpenAI Responses API's response_format parameter
        """
        schema_name = cls._schema_name or cls.__name__.lower()
        schema = cls.model_json_schema()

        if strict:
            schema = make_schema_strict(schema)

        return {
            "type": "json_schema",
            "name": schema_name,
            "strict": strict,
            "schema": schema,
        }

    @classmethod
    def get_schema_name(cls) -> str:
        """Get the schema name used in response_format."""
        return cls._schema_name or cls.__name__.lower()


class _MockFieldInfo:
    """Mock field info for recursive example generation."""

    def __init__(self, ann: Any) -> None:
        self.annotation = ann
        self.examples = None
        self.description = None


def pydantic_to_response_format(
    model: type[BaseModel],
    name: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Convert any Pydantic model to OpenAI Responses API JSON schema format.

    Standalone function that works on any BaseModel (no need to inherit
    from LLMOutputModel).

    Args:
        model: The Pydantic model class.
        name: Schema name (defaults to class name in lowercase).
        strict: If True, makes schema OpenAI-strict-mode compliant.

    Returns:
        Dict suitable for OpenAI Responses API's text.format parameter.
    """
    schema_name = name or model.__name__.lower()
    schema = model.model_json_schema()

    if strict:
        schema = make_schema_strict(schema)

    return {
        "type": "json_schema",
        "name": schema_name,
        "strict": strict,
        "schema": schema,
    }


def make_schema_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively make schema compliant with OpenAI strict mode.

    OpenAI's strict mode requires:
    - additionalProperties: false on all objects
    - all properties listed in 'required' array
    - $ref cannot have sibling keywords (description, etc.)

    Use this function for dynamically-created models (via create_model()).
    For static models, prefer inheriting from LLMOutputModel and using
    to_response_format_flat() which calls this internally.

    Args:
        schema: The JSON schema dict to modify

    Returns:
        Modified schema (modifies in place and returns)
    """
    # Handle $ref - OpenAI strict mode doesn't allow sibling keywords
    if "$ref" in schema:
        ref_value = schema["$ref"]
        schema.clear()
        schema["$ref"] = ref_value
        return schema

    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        # Strict mode requires ALL properties in 'required'
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
            for prop_schema in schema["properties"].values():
                make_schema_strict(prop_schema)
    if "items" in schema:
        make_schema_strict(schema["items"])
    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            make_schema_strict(def_schema)
    # Handle anyOf, oneOf, allOf (used by Pydantic for union types like str | None)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            for sub_schema in schema[key]:
                make_schema_strict(sub_schema)
    return schema


def _get_type_string(annotation: Any) -> str:
    """Convert a type annotation to a human-readable string."""
    if annotation is None:
        return "any"

    origin = get_origin(annotation)

    # Handle Literal types
    if origin is type(None):
        return "null"

    # Handle Optional (Union with None)
    if origin is type(None) or str(annotation).startswith("typing.Optional"):
        args = get_args(annotation)
        inner = args[0] if args else "any"
        return f"{_get_type_string(inner)} | null"

    # Handle Literal
    if str(origin) == "typing.Literal" or str(annotation).startswith("typing.Literal"):
        args = get_args(annotation)
        if args:
            return "enum"
        return "literal"

    # Handle list/List
    if origin is list:
        args = get_args(annotation)
        inner = args[0] if args else "any"
        return f"array[{_get_type_string(inner)}]"

    # Handle dict/Dict
    if origin is dict:
        args = get_args(annotation)
        if len(args) >= 2:
            return f"object<{_get_type_string(args[0])}, {_get_type_string(args[1])}>"
        return "object"

    # Handle Union types (Python 3.10+ | syntax)
    import types

    if isinstance(origin, type) and origin is types.UnionType:
        args = get_args(annotation)
        return " | ".join(_get_type_string(a) for a in args)
    # Also handle typing.Union
    if str(origin) == "typing.Union":
        args = get_args(annotation)
        return " | ".join(_get_type_string(a) for a in args)

    # Handle basic types
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"

    # Handle Pydantic models
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return f"object ({annotation.__name__})"

    # Fallback to string representation
    return str(annotation).replace("typing.", "")


def _get_constraints(field_info: Any) -> str:
    """Extract constraints from field info."""
    constraints = []

    # Check if required
    if field_info.is_required():
        constraints.append("required")
    else:
        constraints.append("optional")

    # Check for min/max length (for lists and strings)
    metadata = field_info.metadata or []
    for m in metadata:
        if hasattr(m, "min_length") and m.min_length is not None:
            constraints.append(f"min: {m.min_length}")
        if hasattr(m, "max_length") and m.max_length is not None:
            constraints.append(f"max: {m.max_length}")
        if hasattr(m, "ge") and m.ge is not None:
            constraints.append(f">= {m.ge}")
        if hasattr(m, "le") and m.le is not None:
            constraints.append(f"<= {m.le}")
        if hasattr(m, "gt") and m.gt is not None:
            constraints.append(f"> {m.gt}")
        if hasattr(m, "lt") and m.lt is not None:
            constraints.append(f"< {m.lt}")

    # Check field_info directly for min/max_length
    if hasattr(field_info, "min_length") and field_info.min_length is not None:
        if f"min: {field_info.min_length}" not in constraints:
            constraints.append(f"min: {field_info.min_length}")
    if hasattr(field_info, "max_length") and field_info.max_length is not None:
        if f"max: {field_info.max_length}" not in constraints:
            constraints.append(f"max: {field_info.max_length}")

    return ", ".join(constraints) if constraints else "required"


def _get_example_value(field_info: Any) -> Any:
    """Generate an example value for a field."""
    annotation = field_info.annotation

    # Check for explicit example in field info
    if hasattr(field_info, "examples") and field_info.examples:
        return field_info.examples[0]

    # Handle None type
    if annotation is None or annotation is type(None):
        return None

    origin = get_origin(annotation)

    # Handle Literal - use first value
    if str(origin) == "typing.Literal" or str(annotation).startswith("typing.Literal"):
        args = get_args(annotation)
        if args:
            return args[0]
        return "value"

    # Handle Optional - recurse into inner type
    if str(annotation).startswith("typing.Optional"):
        args = get_args(annotation)
        if args:
            return _get_example_value(_MockFieldInfo(args[0]))
        return None

    # Handle list
    if origin is list:
        args = get_args(annotation)
        if args:
            inner_example = _get_example_value(_MockFieldInfo(args[0]))
            return [inner_example, inner_example, inner_example]
        return ["item 1", "item 2", "item 3"]

    # Handle dict
    if origin is dict:
        return {"key_1": "value_1", "key_2": "value_2"}

    # Handle basic types with placeholder values
    if annotation is str:
        desc = field_info.description or ""
        if "paragraph" in desc.lower():
            return "Your detailed paragraph here describing the analysis..."
        if "summary" in desc.lower():
            return "Brief summary of the key findings..."
        return "Example string value"

    if annotation is int:
        return 42

    if annotation is float:
        return 3.5

    if annotation is bool:
        return True

    # Handle Pydantic models
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {"nested": "object"}

    return "example_value"


def _get_literal_documentation(cls: type[BaseModel]) -> list[str]:
    """Generate documentation for Literal type fields."""
    lines: list[str] = []

    for field_name, field_info in cls.model_fields.items():
        annotation = field_info.annotation
        origin = get_origin(annotation)

        if str(origin) == "typing.Literal" or str(annotation).startswith("typing.Literal"):
            args = get_args(annotation)
            if args:
                lines.append(f"### `{field_name}` Values")
                lines.append("")
                for val in args:
                    lines.append(f'- **"{val}"**')
                lines.append("")

    return lines
