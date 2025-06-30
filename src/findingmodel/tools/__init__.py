from .add_ids import id_manager
from .create_stub import create_finding_model_stub_from_finding_info, create_model_stub_from_info  # deprecated alias
from .finding_description import (
    # Deprecated aliases
    add_details_to_finding_info,
    add_details_to_info,
    create_finding_info_from_name,
    create_info_from_name,
    describe_finding_name,
    get_detail_on_finding,
)
from .index_codes import add_standard_codes_to_finding_model, add_standard_codes_to_model  # deprecated alias
from .markdown_in import create_finding_model_from_markdown, create_model_from_markdown  # deprecated alias
from .similar_finding_models import find_similar_models

add_ids_to_model = id_manager.add_ids_to_model
add_ids_to_finding_model = id_manager.add_ids_to_finding_model  # deprecated alias

__all__ = [
    "add_details_to_finding_info",
    "add_details_to_info",
    "add_ids_to_finding_model",
    "add_ids_to_model",
    "add_standard_codes_to_finding_model",
    "add_standard_codes_to_model",
    "create_finding_info_from_name",
    "create_finding_model_from_markdown",
    "create_finding_model_stub_from_finding_info",
    "create_info_from_name",
    "create_model_from_markdown",
    "create_model_stub_from_info",
    "describe_finding_name",
    "find_similar_models",
    "get_detail_on_finding",
    "id_manager",
]
