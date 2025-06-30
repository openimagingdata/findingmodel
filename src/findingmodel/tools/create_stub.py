from findingmodel.finding_info import FindingInfo
from findingmodel.finding_model import (
    ChoiceAttribute,
    ChoiceValue,
    FindingModelBase,
)


def create_model_stub_from_info(finding_info: FindingInfo, tags: list[str] | None = None) -> FindingModelBase:
    """
    Create a finding model stub from a FindingInfo object.
    :param finding_info: The FindingInfo object to use for the model.
    :param tags: Optional tags to add to the finding model.
    :return: A FindingModelBase object containing the finding model stub.
    """
    finding_name = finding_info.name.lower()

    def create_presence_element(finding_name: str) -> ChoiceAttribute:
        return ChoiceAttribute(
            name="presence",
            description=f"Presence or absence of {finding_name}",
            values=[
                ChoiceValue(name="absent", description=f"{finding_name.capitalize()} is absent"),
                ChoiceValue(name="present", description=f"{finding_name.capitalize()} is present"),
                ChoiceValue(name="indeterminate", description=f"Presence of {finding_name} cannot be determined"),
                ChoiceValue(name="unknown", description=f"Presence of {finding_name} is unknown"),
            ],
        )

    def create_change_element(finding_name: str) -> ChoiceAttribute:
        return ChoiceAttribute(
            name="change from prior",
            description=f"Whether and how a {finding_name} has changed over time",
            values=[
                ChoiceValue(name="unchanged", description=f"{finding_name.capitalize()} is unchanged"),
                ChoiceValue(name="stable", description=f"{finding_name.capitalize()} is stable"),
                ChoiceValue(name="new", description=f"{finding_name.capitalize()} is new"),
                ChoiceValue(
                    name="resolved", description=f"{finding_name.capitalize()} seen on a prior exam has resolved"
                ),
                ChoiceValue(name="increased", description=f"{finding_name.capitalize()} has increased"),
                ChoiceValue(name="decreased", description=f"{finding_name.capitalize()} has decreased"),
                ChoiceValue(name="larger", description=f"{finding_name.capitalize()} is larger"),
                ChoiceValue(name="smaller", description=f"{finding_name.capitalize()} is smaller"),
            ],
        )

    stub = FindingModelBase(
        name=finding_name,
        description=finding_info.description,
        synonyms=finding_info.synonyms,
        attributes=[
            create_presence_element(finding_name),
            create_change_element(finding_name),
        ],
    )
    if tags:
        stub.tags = tags
    return stub


# Deprecated alias for backward compatibility
def create_finding_model_stub_from_finding_info(
    finding_info: FindingInfo, tags: list[str] | None = None
) -> FindingModelBase:
    """
    DEPRECATED: Use create_model_stub_from_info instead.
    Create a finding model stub from a FindingInfo object.
    """
    import warnings

    warnings.warn(
        "create_finding_model_stub_from_finding_info is deprecated, use create_model_stub_from_info instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_model_stub_from_info(finding_info, tags)
