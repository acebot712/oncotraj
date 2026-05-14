"""Pydantic schemas for the four OncoTraj Parquet tables.

Mirrors DATASET_SPEC.md v1.0.0. This module is the single source of truth for
field names, types, and enum values that parsers and downstream code rely on.

Field coverage is partial in v0.1.0 — patient table is fleshed out enough to
validate; variants/treatments/outcomes are skeletons to be filled per spec.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class SourceDataset(StrEnum):
    GENIE_BPC = "GENIE_BPC"
    MSK_CHORD = "MSK_CHORD"
    TRACERX = "TRACERX"
    FLAURA_SUPP = "FLAURA_SUPP"
    AURA3_SUPP = "AURA3_SUPP"
    ELIOS_SUPP = "ELIOS_SUPP"
    FMI_REGISTRY = "FMI_REGISTRY"
    GUARDANT_REGISTRY = "GUARDANT_REGISTRY"


class Sex(StrEnum):
    male = "male"
    female = "female"
    unknown = "unknown"


class SmokingStatus(StrEnum):
    never = "never"
    former = "former"
    current = "current"
    unknown = "unknown"


class StageAtDiagnosis(StrEnum):
    I = "I"  # noqa: E741
    II = "II"
    IIIA = "IIIA"
    IIIB = "IIIB"
    IIIC = "IIIC"
    IVA = "IVA"
    IVB = "IVB"
    IV_NOS = "IV_NOS"
    unknown = "unknown"


class EgfrVariantClass(StrEnum):
    exon19del = "exon19del"
    L858R = "L858R"
    G719X = "G719X"
    L861Q = "L861Q"
    S768I = "S768I"
    exon20ins = "exon20ins"
    compound_sensitizing = "compound_sensitizing"
    compound_with_resistance = "compound_with_resistance"
    unknown = "unknown"


class ResistanceMechanism(StrEnum):
    """Task C taxonomy (paper outline §5.3)."""

    EGFR_C797S = "EGFR_C797S"
    MET_amplification = "MET_amplification"
    HER2_amplification = "HER2_amplification"
    EGFR_amplification = "EGFR_amplification"
    histologic_transformation = "histologic_transformation"
    other_or_unknown = "other_or_unknown"


class PatientRecord(BaseModel):
    """One row of patients.parquet. See DATASET_SPEC.md §2."""

    patient_id: str = Field(pattern=r"^[A-Z_]+:[A-Za-z0-9_\-]+$")
    source_dataset: SourceDataset
    age_at_diagnosis_years: int = Field(ge=-1, le=110)
    sex: Sex
    smoking_status: SmokingStatus
    diagnosis_date: date
    stage_at_diagnosis: StageAtDiagnosis
    egfr_variant_class: EgfrVariantClass
    site_id: str = Field(min_length=1, max_length=32)
    vital_status_at_last_followup: str
    last_followup_date: date
    included_in_v1_cohort: bool


class VariantRecord(BaseModel):
    """Skeleton — full fields per DATASET_SPEC.md §3 to be added."""

    variant_id: str
    patient_id: str
    sample_id: str


class TreatmentRecord(BaseModel):
    """Skeleton — per DATASET_SPEC.md §4."""

    treatment_id: str
    patient_id: str


class OutcomeRecord(BaseModel):
    """Skeleton — per DATASET_SPEC.md §5."""

    outcome_id: str
    patient_id: str
