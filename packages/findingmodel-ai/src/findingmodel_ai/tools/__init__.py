# AI-powered tools for finding model creation and enrichment
# ruff: noqa: RUF022

# Re-export from findingmodel core for backward compatibility
from findingmodel.create_stub import create_finding_model_stub_from_finding_info, create_model_stub_from_info

from findingmodel_ai.tools.anatomic_location_search import find_anatomic_locations
from findingmodel_ai.tools.finding_description import (
    add_details_to_finding_info,
    add_details_to_info,
    create_finding_info_from_name,
    create_info_from_name,
    describe_finding_name,
    get_detail_on_finding,
)
from findingmodel_ai.tools.finding_enrichment import enrich_finding, enrich_finding_unified
from findingmodel_ai.tools.finding_enrichment_agentic import enrich_finding_agentic
from findingmodel_ai.tools.markdown_in import create_finding_model_from_markdown, create_model_from_markdown
from findingmodel_ai.tools.model_editor import (
    EditResult,
    assign_real_attribute_ids,
    create_edit_agent,
    create_markdown_edit_agent,
    edit_model_markdown,
    edit_model_natural_language,
    export_model_for_editing,
)
from findingmodel_ai.tools.ontology_concept_match import match_ontology_concepts
from findingmodel_ai.tools.similar_finding_models import find_similar_models

__all__ = [
    # Anatomic location search
    "find_anatomic_locations",
    # Model creation
    "create_model_stub_from_info",
    "create_finding_model_stub_from_finding_info",  # deprecated
    # Finding description
    "create_info_from_name",
    "add_details_to_info",
    "create_finding_info_from_name",  # deprecated
    "describe_finding_name",  # deprecated
    "get_detail_on_finding",  # deprecated
    "add_details_to_finding_info",  # deprecated
    # Finding enrichment
    "enrich_finding",
    "enrich_finding_unified",
    "enrich_finding_agentic",
    # Markdown import
    "create_model_from_markdown",
    "create_finding_model_from_markdown",  # deprecated
    # Model editing
    "EditResult",
    "edit_model_natural_language",
    "edit_model_markdown",
    "export_model_for_editing",
    "assign_real_attribute_ids",
    "create_edit_agent",
    "create_markdown_edit_agent",
    # Ontology matching
    "match_ontology_concepts",
    # Similar models
    "find_similar_models",
]
