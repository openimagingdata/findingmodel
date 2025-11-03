from dataclasses import dataclass
from typing import Mapping, cast

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.agent import Agent

from findingmodel import Index
from findingmodel.config import settings
from findingmodel.finding_model import FindingModelFull
from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID
from findingmodel.tools.common import get_openai_model

# Module-level Index instance with lazy initialization
_index: Index | None = None


def _get_index() -> Index:
    """Get or create module-level Index instance."""
    global _index
    if _index is None:
        _index = Index()
    return _index


class EditResult(BaseModel):
    model: FindingModelFull = Field(description="Updated FindingModelFull after applying edits")
    rejections: list[str] = Field(default_factory=list, description="List of reasons for any rejected edit requests")
    changes: list[str] = Field(
        default_factory=list,
        description="Human-readable summaries of the successful changes applied to the model",
    )


@dataclass
class EditDeps:
    original: FindingModelFull
    requested_text: str


def _combine_instruction_sections(*sections: str) -> str:
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _common_editing_instructions() -> str:
    return _combine_instruction_sections(
        "You are an expert medical imaging finding model editor.",
        "Core mission:\n"
        "- Update a FindingModelFull JSON while preserving existing clinical meaning.\n"
        "- Apply only safe additions and non-semantic enhancements; never remove or rename existing attributes or values.\n"
        "- Leave any requested change untouched and document a rejection when it would alter semantics or violate safety rules.",
        "ID handling:\n"
        "- Preserve ALL existing OIFM IDs (model, attributes, values, codes); they are immutable.\n"
        f'- When adding new attributes, assign the literal ID "{PLACEHOLDER_ATTRIBUTE_ID}".\n'
        "- ALWAYS keep that literal ID for every brand-new attribute even if the request suggests another value.\n"
        "- When adding new choice values, derive value_code entries from the attribute ID using '.n' ordering starting at 0 and do not change existing value codes.",
        "Content quality and enrichment:\n"
        "- Generate concise, clinically appropriate descriptions, synonyms, tags, units, or ranges when the user omits them; do not reject changes solely because metadata was missing.\n"
        "- Write attribute descriptions as one sentence (two only when medically necessary).\n"
        "- Write choice value descriptions as a single sentence or short clause.\n"
        "- Correct spelling, grammar, and capitalization; expand acronyms in attribute names and expand any acronym the first time it appears in a description.\n"
        "- Keep attribute and value names lowercase unless they are proper nouns (e.g., Crohn's disease).\n"
        "- Interpret numeric hints such as min/max/unit text and populate structured fields even when punctuation is sparse.",
        "Change tracking and rejections:\n"
        "- Populate the 'changes' array with past-tense bullet-style summaries for every successful addition or textual update; leave it empty only when the model is unchanged.\n"
        "- Use the 'rejections' array only for portions you could not safely execute, and provide a clear reason for each entry.\n"
        "- Do not invent new policy requirements (e.g., 'mandated metadata'); only reject when the request truly conflicts with these rules or remains ambiguous after reasonable interpretation.",
    )


def _natural_language_instructions() -> str:
    return _combine_instruction_sections(
        "Input channel: natural-language command.",
        "- Interpret informal wording generously and infer implied attributes, values, or metadata when the intent is clear.\n"
        "- When ambiguity remains after your best effort, keep the affected portion unchanged and record a rejection describing the uncertainty.",
    )


def _edited_text_instructions() -> str:
    return _combine_instruction_sections(
        "Input channel: an edited text representation of the model (often exported as Markdown) that may contain inconsistent or missing formatting.",
        "- Treat headings, bullet-like lines, indentation, and blank lines as loose hints; interpret them even when Markdown syntax is imperfect or absent.\n"
        "- Infer new attributes or choice values from grouped lines in the order they appear, correcting spelling and casing according to the content quality rules.\n"
        "- When numeric snippets mention min/max/unit or ranges, treat them as numeric metadata regardless of capitalization or punctuation.\n"
        "- If the edited text omits any original attribute or value (even entire sections), preserve the original content exactly as it appears in the provided JSON (no new metadata or text) and add clear rejection messages explaining that deletions were ignored.",
    )


def create_edit_agent() -> Agent[EditDeps, EditResult]:
    """Factory for the natural-language editing agent.

    Exposed to facilitate testing with TestModel/FunctionModel via agent.override(...).
    """
    instructions = _combine_instruction_sections(
        _common_editing_instructions(),
        _natural_language_instructions(),
        "Output strictly according to the provided schema with the updated FindingModelFull JSON, the list of rejections, and the list of changes you applied.",
    )

    agent = Agent[EditDeps, EditResult](
        model=get_openai_model(settings.openai_default_model),
        deps_type=EditDeps,
        output_type=EditResult,
        instructions=instructions,
    )

    @agent.output_validator
    def _validate_output(ctx: RunContext[EditDeps], output: EditResult) -> EditResult:
        _normalize_new_attribute_ids(original=ctx.deps.original, updated=output.model)
        errors = _basic_edit_validation(original=ctx.deps.original, updated=output.model)
        if errors:
            raise ModelRetry(
                "Invalid edits: "
                + "; ".join(errors)
                + "\nCorrect the output by preserving all existing IDs and attributes/values; only add safe items."
            )
        return output

    return agent


def create_markdown_edit_agent() -> Agent[EditDeps, EditResult]:
    """Factory for the text-editing agent used for Markdown-like edits.

    The agent receives the current FindingModelFull (JSON) and an edited text string
    (often exported as Markdown) representing desired changes. It must output a COMPLETE
    FindingModelFull JSON while:
      - Preserving ALL existing OIFM IDs (model, attributes, values, codes)
      - Applying ONLY safe changes (add attributes/values; add non-semantic text like descriptions/synonyms/tags)
      - Never removing or renaming existing attributes/values or changing existing semantics
    """
    instructions = _combine_instruction_sections(
        _common_editing_instructions(),
        _edited_text_instructions(),
        "Output strictly according to the provided schema with the updated FindingModelFull JSON, the list of rejections, and the list of changes you applied.",
    )

    agent = Agent[EditDeps, EditResult](
        model=get_openai_model(settings.openai_default_model),
        deps_type=EditDeps,
        output_type=EditResult,
        instructions=instructions,
    )

    @agent.output_validator
    def _validate_output(ctx: RunContext[EditDeps], output: EditResult) -> EditResult:
        _normalize_new_attribute_ids(original=ctx.deps.original, updated=output.model)
        errors = _basic_edit_validation(original=ctx.deps.original, updated=output.model)
        if errors:
            raise ModelRetry(
                "Invalid edits: "
                + "; ".join(errors)
                + "\nCorrect the output by preserving all existing IDs and attributes/values; only add safe items."
            )
        return output

    return agent


async def edit_model_natural_language(
    model: FindingModelFull, command: str, *, agent: Agent[EditDeps, EditResult] | None = None
) -> EditResult:
    """
    Edit a FindingModelFull using a natural language command via an LLM agent.
    Only allow safe additions and non-semantic edits; preserve ALL existing OIFM IDs.
    """
    agent = agent or create_edit_agent()

    # Provide the current model JSON and the user's command as the run input
    prompt = (
        "You will receive the current FindingModelFull JSON followed by a natural-language editing command.\n"
        "Interpret the command generously and perform any safe additions or non-semantic text improvements implied by the request, even when details are sparse.\n"
        "When information such as descriptions, synonyms, numeric ranges, or units is missing, supply concise, clinically appropriate text yourself.\n"
        "If a portion of the command would remove or rename existing content, or remains ambiguous after reasonable interpretation, leave that part unchanged and add a rejection explaining why it could not be applied.\n"
        "Only include entries in 'rejections' for requested edits you could not safely apply.\n"
        "Summarize each successful change you applied in the 'changes' array using concise past-tense phrases.\n"
        f"Current model JSON:\n```json\n{model.model_dump_json()}\n```\n\n"
        f"Command: {command}\n"
    )
    try:
        result = await agent.run(prompt, deps=EditDeps(original=model, requested_text=command))
        return result.output
    except Exception:
        # Graceful fallback: if the agent run fails for any reason, return the original model unchanged
        return EditResult(model=model, rejections=["Agent run failed; no changes applied."], changes=[])


def _render_top_metadata_lines(model: FindingModelFull) -> list[str]:
    """Render the top metadata section of the editable Markdown.

    Format expected by tests:
    - Title: "# {Name capitalized}"
    - Optional description paragraph
    - Optional "Synonyms: ..." and "Tags: ..." lines
    - Blank line and then "## Attributes"
    """
    lines: list[str] = []
    lines.append(f"# {model.name.capitalize()}")
    lines.append("")
    # Description paragraph (when present)
    if model.description:
        lines.append(model.description)
        lines.append("")
    if model.synonyms:
        lines.append(f"Synonyms: {', '.join(model.synonyms)}")
    if model.tags:
        lines.append(f"Tags: {', '.join(model.tags)}")
    if model.synonyms or model.tags:
        lines.append("")
    lines.append("## Attributes")
    lines.append("")
    return lines


def _render_attribute_lines(attr: object) -> list[str]:
    """Render a single attribute in the simple Markdown format expected by tests."""
    lines: list[str] = []
    name = cast(str, getattr(attr, "name", "")).strip()
    atype = cast(str, getattr(attr, "type", "")).strip()
    description = cast(str | None, getattr(attr, "description", None))

    # Attribute header uses raw name (no capitalization)
    lines.append(f"### {name}")
    lines.append("")
    if description:
        lines.append(description)
        lines.append("")

    if atype == "choice":
        values = cast(list[object], getattr(attr, "values", []))
        for v in values:
            vname = cast(str, getattr(v, "name", "")).strip()
            vdesc = cast(str | None, getattr(v, "description", None))
            if vdesc:
                lines.append(f"- {vname}: {vdesc}")
            else:
                lines.append(f"- {vname}")
    elif atype == "numeric":
        parts: list[str] = []
        minimum = getattr(attr, "minimum", None)
        maximum = getattr(attr, "maximum", None)
        unit = cast(str | None, getattr(attr, "unit", None))
        if minimum is not None:
            parts.append(f"min {minimum}")
        if maximum is not None:
            parts.append(f"max {maximum}")
        if unit:
            parts.append(f"unit {unit}")
        if parts:
            lines.append("- " + "; ".join(parts))

    lines.append("")
    return lines


def export_model_for_editing(model: FindingModelFull, *, attributes_only: bool = False) -> str:
    """Export a model to a human-editable Markdown format used for round-trip editing in tests.

    - When attributes_only is True, omit the top metadata and the "## Attributes" header.
    - Attribute names are not capitalized; value descriptions add a colon only when present.
    - Numeric attributes are summarized in a single bullet: "- min X; max Y; unit Z" with only present parts.
    """
    lines: list[str] = []
    if not attributes_only:
        lines.extend(_render_top_metadata_lines(model))
    for attr in model.attributes:
        lines.extend(_render_attribute_lines(attr))
    # Ensure trailing newline consistency
    md = "\n".join(lines).rstrip() + "\n"
    return md


async def edit_model_markdown(
    model: FindingModelFull,
    edited_markdown: str,
    *,
    agent: Agent[EditDeps, EditResult] | None = None,
) -> EditResult:
    """Edit a FindingModelFull given an edited text string via an LLM agent.

    The agent is instructed to apply only safe changes, tolerate loosely formatted Markdown-like input,
    preserve IDs, and output a complete model JSON. Rejections are summarized by a secondary LLM call
    (when enabled) comparing original vs updated with the request.
    """
    agent = agent or create_markdown_edit_agent()

    attribute_reference_lines = "\n".join(
        f"- {cast(str, getattr(attr, 'name', '')).strip()} ({getattr(attr, 'oifma_id', '<missing id>')})"
        for attr in model.attributes
        if cast(str, getattr(attr, "name", "")).strip()
    )
    attribute_reference = (
        "Compare the edited text against the following original attribute sections. If any of these names are absent or truncated in the edited text, treat it as an attempted deletion: keep the entire original attribute (including descriptions, values, and metadata) exactly as shown in the JSON output and add a rejection about the attempted deletion.\n"
        "Original attributes (name and OIFMA ID):\n"
        f"{attribute_reference_lines}\n\n"
        if attribute_reference_lines
        else ""
    )

    prompt = (
        "You will receive the current FindingModelFull JSON and an edited text representation of the model.\n"
        "Treat the edited text as loose, user-authored content that may resemble Markdown but might omit proper syntax or punctuation.\n"
        "Infer safe additions and non-semantic text improvements from the edited text, supplying concise clinical descriptions, synonyms, numeric ranges, and units whenever the user leaves them blank.\n"
        "If the edited text attempts to remove or rename existing content, keep those items unchanged in the JSON output and record a rejection explaining the forbidden deletion.\n"
        "When lines suggest new attributes or choice values, add them in the presented order, correcting spelling and casing according to the quality rules.\n"
        "Interpret any mention of minimums, maximums, units, or numeric ranges as numeric metadata even when formatting is inconsistent. Populate the structured fields accordingly.\n"
        "Only include entries in 'rejections' for requested edits you could not safely apply.\n"
        "Summarize the applied changes in the 'changes' array using short past-tense phrases; leave it empty only if no updates were required.\n"
        "Return the COMPLETE updated FindingModelFull as strict JSON, with the accompanying rejections and changes arrays as specified in the schema.\n\n"
        f"{attribute_reference}"
        f"Current model JSON:\n```\n{model.model_dump_json()}\n```\n\n"
        f"Edited text:\n```\n{edited_markdown}\n```\n"
    )
    try:
        result = await agent.run(prompt, deps=EditDeps(original=model, requested_text=edited_markdown))
        return result.output
    except Exception:
        # Graceful fallback: return the original model unchanged
        return EditResult(model=model, rejections=["Agent run failed; no changes applied."], changes=[])


def _basic_edit_validation(*, original: FindingModelFull, updated: FindingModelFull) -> list[str]:
    """Compose simple validators and return combined messages (deduped)."""
    orig_attrs = {getattr(a, "oifma_id", None): a for a in original.attributes}
    upd_attrs = {getattr(a, "oifma_id", None): a for a in updated.attributes}
    parts: list[list[str]] = [
        _validate_model_id(original, updated),
        _validate_original_attributes_present(orig_attrs, upd_attrs),
        _validate_new_items_have_ids(orig_attrs, upd_attrs),
        _validate_choice_value_indices_and_bases(orig_attrs, upd_attrs),
    ]
    flat = [m for sub in parts for m in sub]
    return list(dict.fromkeys(flat))


def _normalize_new_attribute_ids(*, original: FindingModelFull, updated: FindingModelFull) -> None:
    """Force placeholder IDs for newly added attributes and their values."""

    original_ids = {getattr(attr, "oifma_id", None) for attr in original.attributes}
    normalized_attrs = []

    for attr in updated.attributes:
        oifma_id = getattr(attr, "oifma_id", None)
        if oifma_id not in original_ids:
            value_updates: dict[str, object] = {"oifma_id": PLACEHOLDER_ATTRIBUTE_ID}
            if getattr(attr, "type", None) == "choice":
                normalized_values = []
                for idx, value in enumerate(getattr(attr, "values", []) or []):
                    normalized_values.append(
                        value.model_copy(update={"value_code": f"{PLACEHOLDER_ATTRIBUTE_ID}.{idx}"})
                    )
                value_updates["values"] = normalized_values
            attr = attr.model_copy(update=value_updates)
        normalized_attrs.append(attr)

    updated.attributes = normalized_attrs


def assign_real_attribute_ids(
    model: FindingModelFull,
    *,
    source: str | None = None,
    index: Index | None = None,
) -> FindingModelFull:
    """Replace placeholder attribute IDs using Index database queries.

    Args:
        model: Model with potential placeholder attribute IDs.
        source: Source code (3-4 uppercase letters). When omitted, inferred from the model's OIFM ID.
        index: Index instance to use for ID generation. When omitted, uses the module-level instance.

    Returns:
        Model with all placeholder IDs replaced by unique attribute IDs.
    """
    idx = index if index is not None else _get_index()
    return idx.finalize_placeholder_attribute_ids(model, source=source)


def _validate_model_id(original: FindingModelFull, updated: FindingModelFull) -> list[str]:
    return (
        [f"Model ID changed from {original.oifm_id} to {updated.oifm_id} (not allowed)"]
        if original.oifm_id != updated.oifm_id
        else []
    )


def _validate_original_attributes_present(
    orig_attrs: Mapping[str | None, object], upd_attrs: Mapping[str | None, object]
) -> list[str]:
    errors: list[str] = []
    for oifma_id, a in orig_attrs.items():
        if oifma_id is None:
            name = getattr(a, "name", "<unknown>")
            errors.append(f"Original attribute '{name}' missing ID; cannot validate presence")
            continue
        if oifma_id not in upd_attrs:
            name = getattr(a, "name", oifma_id)
            errors.append(f"Missing original attribute '{name}' ({oifma_id}) in updated model")
            continue
        # For choice attributes, ensure all original values are present
        a_upd = upd_attrs[oifma_id]
        if getattr(a, "type", None) == "choice":
            orig_vals = {getattr(v, "value_code", None): v for v in getattr(a, "values", [])}
            upd_vals = {getattr(v, "value_code", None): v for v in getattr(a_upd, "values", [])}
            for vcode, v in orig_vals.items():
                if vcode is None:
                    vname = getattr(v, "name", "<unknown>")
                    errors.append(
                        f"Original choice value '{vname}' in attribute ({oifma_id}) missing code; cannot validate presence"
                    )
                    continue
                if vcode not in upd_vals:
                    vname = getattr(v, "name", vcode)
                    aname = getattr(a, "name", oifma_id)
                    errors.append(f"Missing original value '{vname}' ({vcode}) in attribute '{aname}' ({oifma_id})")
    return errors


def _validate_new_items_have_ids(
    orig_attrs: Mapping[str | None, object], upd_attrs: Mapping[str | None, object]
) -> list[str]:
    errors: list[str] = []
    for oifma_id, a in upd_attrs.items():
        if oifma_id not in orig_attrs:
            name = getattr(a, "name", "<unknown>")
            # New attributes must use the placeholder literal ID exactly
            if oifma_id != PLACEHOLDER_ATTRIBUTE_ID:
                errors.append(
                    f"New attribute '{name}' must use the literal ID {PLACEHOLDER_ATTRIBUTE_ID} (got {oifma_id!r})"
                )
            if getattr(a, "type", None) == "choice":
                for v in getattr(a, "values", []) or []:
                    vcode = getattr(v, "value_code", None)
                    vname = getattr(v, "name", "<unknown>")
                    # New choice value codes must derive from the literal attribute ID with ".n" appended
                    if not isinstance(vcode, str) or not vcode.startswith(f"{PLACEHOLDER_ATTRIBUTE_ID}."):
                        errors.append(f"New choice value '{vname}' in attribute '{name}' missing or invalid value_code")
    return errors


def _validate_choice_value_indices_and_bases(
    orig_attrs: Mapping[str | None, object], upd_attrs: Mapping[str | None, object]
) -> list[str]:
    """Top-level dispatcher across updated choice attributes."""
    errors: list[str] = []

    for oifma_id, upd_attr in upd_attrs.items():
        if getattr(upd_attr, "type", None) != "choice":
            continue
        raw_name = getattr(upd_attr, "name", None)
        name = raw_name if isinstance(raw_name, str) and raw_name else (oifma_id or "<unknown>")

        vcodes, suffixes, base_errors, values_len = _collect_value_codes(oifma_id, upd_attr, name)
        errors.extend(base_errors)

        if oifma_id not in orig_attrs:
            errors.extend(_validate_new_attr_suffixes(name, oifma_id, suffixes, values_len))
        else:
            orig_attr = orig_attrs[oifma_id]
            errors.extend(_validate_existing_attr_suffixes(name, oifma_id, vcodes, suffixes, orig_attr))

    return errors


def _parse_suffix(vcode: str) -> int | None:
    try:
        return int(vcode.rsplit(".", 1)[1])
    except Exception:
        return None


def _collect_value_codes(
    oifma_id: str | None, upd_attr: object, name: str
) -> tuple[list[str], list[int], list[str], int]:
    """Gather value_code strings and integer suffixes; validate base and duplicates.

    Returns: (vcodes, suffixes, errors, values_len)
    """
    errors: list[str] = []
    values = list(getattr(upd_attr, "values", []) or [])
    vcodes: list[str] = []
    suffixes: list[int] = []
    base = f"{oifma_id}." if oifma_id else None

    for v in values:
        vcode = getattr(v, "value_code", None)
        vname = getattr(v, "name", "<unknown>")
        if not isinstance(vcode, str):
            errors.append(f"Choice value '{vname}' in attribute '{name}' ({oifma_id}) missing value_code string")
            continue
        if base and not vcode.startswith(base):
            errors.append(
                f"Choice value '{vname}' in attribute '{name}' has value_code base '{vcode.split('.')[0]}' not matching attribute ID '{oifma_id}'"
            )
        vcodes.append(vcode)
        s = _parse_suffix(vcode)
        if s is None:
            errors.append(
                f"Attribute '{name}' ({oifma_id}) has value_code '{vcode}' without a valid numeric '.n' suffix"
            )
        else:
            suffixes.append(s)

    if len(vcodes) != len(set(vcodes)):
        errors.append(f"Attribute '{name}' ({oifma_id}) has duplicate value_code entries")

    return vcodes, suffixes, errors, len(values)


def _validate_new_attr_suffixes(name: str, oifma_id: str | None, suffixes: list[int], values_len: int) -> list[str]:
    if oifma_id is None:
        return [f"New attribute '{name}' missing oifma_id"]
    expected = set(range(values_len))
    got = set(suffixes)
    return (
        [f"New attribute '{name}' value_code suffixes must be 0..{values_len - 1} (got {sorted(got)})"]
        if got != expected
        else []
    )


def _validate_existing_attr_suffixes(
    name: str, oifma_id: str | None, vcodes: list[str], suffixes: list[int], orig_attr: object
) -> list[str]:
    if oifma_id is None:
        return [f"Existing attribute '{name}' missing oifma_id"]
    errors: list[str] = []
    orig_values = list(getattr(orig_attr, "values", []) or [])
    orig_vcodes = {getattr(v, "value_code", None) for v in orig_values}
    orig_suffixes = {_parse_suffix(c) for c in orig_vcodes if isinstance(c, str) and c.startswith(f"{oifma_id}.")}
    orig_suffixes_int = {s for s in orig_suffixes if isinstance(s, int)}

    new_vcodes = {vc for vc in vcodes if vc not in orig_vcodes}
    new_suffixes = {_parse_suffix(vc) for vc in new_vcodes}
    new_suffixes_int = {s for s in new_suffixes if isinstance(s, int)}

    # Collision check
    colliding = sorted(orig_suffixes_int & new_suffixes_int)
    if colliding:
        errors.append(
            f"Attribute '{name}' ({oifma_id}) has new value_code suffixes colliding with existing ones: {colliding}"
        )

    if new_suffixes_int:
        start = (max(orig_suffixes_int) + 1) if orig_suffixes_int else 0
        expected_contiguous = set(range(start, start + len(new_suffixes_int)))
        if new_suffixes_int != expected_contiguous:
            errors.append(
                f"Attribute '{name}' ({oifma_id}) must assign new value_code suffixes contiguously starting at {start} (got {sorted(new_suffixes_int)})"
            )

    return errors
