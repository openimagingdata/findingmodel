"""Canonical metadata assignment APIs."""

from findingmodel_ai.metadata.assignment import (
    MetadataAssignmentDecision,
    assign_metadata,
    create_entity_type_agent,
    create_etiology_tempo_agent,
    create_modality_applicability_agent,
    create_subspecialty_domain_agent,
)
from findingmodel_ai.metadata.auditor import (
    EnrichmentAuditFlag,
    EnrichmentAuditResult,
    audit_enrichment,
    create_enrichment_auditor_agent,
)
from findingmodel_ai.metadata.decisions import (
    EntityTypeDecision,
    EtiologyTempoDecision,
    ModalityApplicabilityDecision,
    OntologyCandidateDecision,
    SubspecialtyDomainDecision,
)
from findingmodel_ai.metadata.ontology_cache import OntologyEvidenceUsage, OntologyLookupCache, OntologyLookupEvidence
from findingmodel_ai.metadata.types import (
    AnatomicCandidate,
    ConfidenceFieldKey,
    FieldConfidence,
    MetadataAssignmentResult,
    MetadataAssignmentReview,
    OntologyCandidate,
    OntologyCandidateRejectionReason,
    OntologyCandidateRelationship,
    OntologyCandidateReport,
)

__all__ = [
    "AnatomicCandidate",
    "ConfidenceFieldKey",
    "EnrichmentAuditFlag",
    "EnrichmentAuditResult",
    "EntityTypeDecision",
    "EtiologyTempoDecision",
    "FieldConfidence",
    "MetadataAssignmentDecision",
    "MetadataAssignmentResult",
    "MetadataAssignmentReview",
    "ModalityApplicabilityDecision",
    "OntologyCandidate",
    "OntologyCandidateDecision",
    "OntologyCandidateRejectionReason",
    "OntologyCandidateRelationship",
    "OntologyCandidateReport",
    "OntologyEvidenceUsage",
    "OntologyLookupCache",
    "OntologyLookupEvidence",
    "SubspecialtyDomainDecision",
    "assign_metadata",
    "audit_enrichment",
    "create_enrichment_auditor_agent",
    "create_entity_type_agent",
    "create_etiology_tempo_agent",
    "create_modality_applicability_agent",
    "create_subspecialty_domain_agent",
]
