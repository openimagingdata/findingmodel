# Import types for add_ids_to_model functions
from findingmodel.finding_model import FindingModelBase, FindingModelFull
from findingmodel.index import DuckDBIndex as Index

# Import non-AI utility tools from local package
from .index_codes import add_standard_codes_to_finding_model, add_standard_codes_to_model  # deprecated alias

# NOTE: AI tools have moved to findingmodel_ai package.
# Import from the appropriate subpackage:
#   - findingmodel_ai.enrichment (enrich_finding, enrich_finding_unified, enrich_finding_agentic)
#   - findingmodel_ai.search (find_anatomic_locations, find_similar_models, match_ontology_concepts)
#   - findingmodel_ai.authoring (create_info_from_name, create_model_from_markdown, edit_model_natural_language)


def add_ids_to_model(
    finding_model: FindingModelBase | FindingModelFull,
    source: str,
) -> FindingModelFull:
    """Generate and add IDs to a finding model using database-based ID generation.

    Replaces GitHub-based IdManager with Index database queries.

    Args:
        finding_model: Model to add IDs to (base or full).
        source: 3-4 uppercase letter source code.

    Returns:
        FindingModelFull with all IDs generated.

    Example:
        >>> from findingmodel.tools import add_ids_to_model
        >>> model = add_ids_to_model(base_model, "GMTS")
        >>> print(model.oifm_id)  # "OIFM_GMTS_472951"
    """
    index = Index()
    return index.add_ids_to_model(finding_model, source)


def add_ids_to_finding_model(
    finding_model: FindingModelBase | FindingModelFull,
    source: str,
) -> FindingModelFull:
    """DEPRECATED: Use add_ids_to_model instead.

    Generate and add IDs to a finding model.

    Args:
        finding_model: Model to add IDs to (base or full).
        source: 3-4 uppercase letter source code.

    Returns:
        FindingModelFull with all IDs generated.
    """
    import warnings

    warnings.warn(
        "add_ids_to_finding_model is deprecated, use add_ids_to_model instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return add_ids_to_model(finding_model, source)


__all__ = [
    "add_ids_to_finding_model",
    "add_ids_to_model",
    "add_standard_codes_to_finding_model",
    "add_standard_codes_to_model",
]

# AI tools moved to findingmodel_ai - removed from __all__:
# - add_details_to_finding_info, add_details_to_info
# - create_finding_info_from_name, create_info_from_name
# - create_finding_model_from_markdown, create_model_from_markdown
# - create_finding_model_stub_from_finding_info, create_model_stub_from_info
# - describe_finding_name, get_detail_on_finding
# - find_anatomic_locations
# - find_similar_models
