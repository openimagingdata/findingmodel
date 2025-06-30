from typing import Protocol, Sequence

from findingmodel.finding_model import ChoiceAttributeIded, FindingModelFull, IndexCodeList
from findingmodel.index_code import IndexCode


class Codeable(Protocol):
    index_codes: IndexCodeList | None


def _add_index_codes(target: Codeable, name: str) -> None:
    found_codes = STANDARD_CODES.get(name)
    if not found_codes or not isinstance(found_codes, Sequence) or len(found_codes) == 0:
        return
    if target.index_codes is None:
        target.index_codes = []
    assert isinstance(target.index_codes, list)
    have_codes = {f"{code.system}:{code.code}" for code in target.index_codes}
    for code in found_codes:
        if f"{code.system}:{code.code}" in have_codes:
            continue
        target.index_codes.append(code)
        have_codes.add(f"{code.system}:{code.code}")


def add_standard_codes_to_model(finding_model: FindingModelFull) -> None:
    """
    Add standard codes to the finding model.
    :param finding_model: The finding model to add standard codes to.
    :return: None
    """
    for attribute in finding_model.attributes:
        _add_index_codes(attribute, attribute.name.lower())
        if isinstance(attribute, ChoiceAttributeIded):
            for value in attribute.values:
                _add_index_codes(value, value.name.lower())


# Deprecated alias for backward compatibility
def add_standard_codes_to_finding_model(finding_model: FindingModelFull) -> None:
    """
    DEPRECATED: Use add_standard_codes_to_model instead.
    Add standard codes to the finding model.
    """
    import warnings

    warnings.warn(
        "add_standard_codes_to_finding_model is deprecated, use add_standard_codes_to_model instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return add_standard_codes_to_model(finding_model)


STANDARD_CODES = {
    "presence": [IndexCode(system="SNOMED", code="705057003", display="Presence (property) (qualifier value)")],
    "absent": [
        IndexCode(system="RADLEX", code="RID28473", display="absent"),
        IndexCode(system="SNOMED", code="2667000", display="Absent (qualifier value)"),
    ],
    "present": [
        IndexCode(system="RADLEX", code="RID28472", display="present"),
        IndexCode(system="SNOMED", code="52101004", display="Present (qualifier value)"),
    ],
    "indeterminate": [
        IndexCode(system="RADLEX", code="RID39110", display="indeterminate"),
        IndexCode(system="SNOMED", code="82334004", display="Indeterminate (qualifier value)"),
    ],
    "unknown": [
        IndexCode(system="RADLEX", code="RID5655", display="unknown"),
        IndexCode(system="SNOMED", code="261665006", display="Unknown (qualifier value)"),
    ],
    "change from prior": [
        IndexCode(system="RADLEX", code="RID49896", display="change"),
        IndexCode(system="SNOMED", code="263703002", display="Changed status (qualifier value)"),
    ],
    "stable": [
        IndexCode(system="RADLEX", code="RID5734", display="stable"),
        IndexCode(system="SNOMED", code="58158008", display="Stable (qualifier value)"),
    ],
    "unchanged": [
        IndexCode(system="RADLEX", code="RID39268", display="unchanged"),
        IndexCode(system="SNOMED", code="260388006", display="No status change (qualifier value)"),
    ],
    "increased": [
        IndexCode(system="RADLEX", code="RID36043", display="increased"),
        IndexCode(system="SNOMED", code="35105006", display="Increased (qualifier value)"),
    ],
    "decreased": [
        IndexCode(system="RADLEX", code="RID36044", display="decreased"),
        IndexCode(system="SNOMED", code="1250004", display="Decreased (qualifier value)"),
    ],
    "new": [
        IndexCode(system="RADLEX", code="RID5720", display="new"),
        IndexCode(system="SNOMED", code="7147002", display="New (qualifier value)"),
    ],
    "quantity": [
        IndexCode(system="RADLEX", code="RID5761", display="quantity descriptor"),
        IndexCode(system="SNOMED", code="246205007", display="Quantity (attribute)"),
    ],
    "multiple": [
        IndexCode(system="RADLEX", code="RID5765", display="multiple"),
        IndexCode(system="SNOMED", code="255204007", display="Multiple (qualifier value)"),
    ],
    "single": [
        IndexCode(system="RADLEX", code="RID5762", display="single"),
        IndexCode(system="SNOMED", code="50607009", display="Singular (qualifier value)"),
    ],
    "severity": [IndexCode(system="SNOMED", code="246112005", display="Severity (attribute)")],
    "mild": [
        IndexCode(system="RADLEX", code="RID5671", display="mild"),
        IndexCode(system="SNOMED", code="255604002", display="Mild (qualifier value)"),
    ],
    "moderate": [
        IndexCode(system="RADLEX", code="RID5672", display="moderate"),
        IndexCode(system="SNOMED", code="1255665007", display="Moderate (qualifier value)"),
    ],
    "severe": [
        IndexCode(system="RADLEX", code="RID5673", display="severe"),
        IndexCode(system="SNOMED", code="24484000", display="Severe (severity modifier) (qualifier value)"),
    ],
    "location": [
        IndexCode(system="RADLEX", code="RID39038", display="location"),
        IndexCode(system="SNOMED", code="758637006", display="Anatomic location (property) (qualifier value)"),
    ],
    "size": [
        IndexCode(system="SNOMED", code="246115007", display="Size (attribute)"),
        IndexCode(system="RADLEX", code="RID5772", display="size descriptor"),
    ],
    "larger": [
        IndexCode(system="RADLEX", code="RID5791", display="enlarged"),
        IndexCode(system="SNOMED", code="263768009", display="Greater (qualifier value)"),
    ],
    "smaller": [
        IndexCode(system="RADLEX", code="RID38669", display="diminished"),
        IndexCode(system="SNOMED", code="263796003", display="Lesser (qualifier value)"),
    ],
}
