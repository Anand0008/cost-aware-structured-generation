"""
============================================================================
JSON VALIDATOR UTILITY
============================================================================
Purpose: Validate model responses against tier 1-4 schema and auto-fix common issues
Features:
    - Complete schema validation for tier_0 through tier_4
    - Nested structure validation
    - Array element validation
    - Auto-fix common formatting errors
    - Field type validation with ranges
    - Required field checking
    - Detailed error reporting with context

Usage:
    from utils.json_validator import validate_complete_schema, auto_fix_json
    
    # Validate complete JSON
    is_valid, errors = validate_complete_schema(model_response)
    
    # Auto-fix common issues
    fixed_json = auto_fix_json(model_response)
    
    # Validate specific tier
    errors = validate_tier_1(tier_1_data)

Author: GATE AE SOTA Pipeline
============================================================================
"""

import json
from typing import Dict, Any, List, Tuple, Optional, Union

from pipeline.utils.logging_utils import setup_logger

logger = setup_logger("json_validator")


# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

TIER_0_SCHEMA = {
    "required_fields": [
        "content_type",
        "media_type",
        "difficulty_score",
        "complexity_flags",
        "use_gpt51",
        "classification_confidence",
        "classification_reasoning",
        "weight_strategy"
    ],
    "field_types": {
        "content_type": str,
        "media_type": str,
        "difficulty_score": int,
        "complexity_flags": dict,
        "use_gpt51": bool,
        "classification_confidence": float,
        "classification_reasoning": str,
        "weight_strategy": str
    },
    "valid_values": {
        "content_type": [
            "conceptual_theory",
            "mathematical_derivation",
            "numerical_calculation",
            "conceptual_application"
        ],
        "media_type": ["text_only", "image_based"],
        "weight_strategy": [
            "BALANCED",
            "CONCEPTUAL_WEIGHTED",
            "MATH_WEIGHTED",
            "NUMERICAL_WEIGHTED",
            "APPLICATION_WEIGHTED",
            "VISION_WEIGHTED",
            "MATH_IMAGE_HYBRID"
        ]
    },
    "value_ranges": {
        "difficulty_score": (1, 10),
        "classification_confidence": (0.0, 1.0)
    },
    "nested": {
        "complexity_flags": {
            "required_fields": [
                "requires_derivation",
                "multi_concept_integration",
                "ambiguous_wording",
                "image_interpretation_complex",
                "edge_case_scenario",
                "multi_step_reasoning",
                "approximation_needed"
            ],
            "field_types": {
                "requires_derivation": bool,
                "multi_concept_integration": bool,
                "ambiguous_wording": bool,
                "image_interpretation_complex": bool,
                "edge_case_scenario": bool,
                "multi_step_reasoning": bool,
                "approximation_needed": bool
            }
        }
    }
}

TIER_1_SCHEMA = {
    "required_fields": [
        "answer_validation",
        "explanation",
        "hierarchical_tags",
        "prerequisites",
        "difficulty_analysis",
        "textbook_references",
        "video_references",
        "step_by_step_solution",
        "formulas_principles",
        "real_world_applications"
    ],
    "field_types": {
        "answer_validation": dict,
        "explanation": dict,
        "hierarchical_tags": dict,
        "prerequisites": dict,
        "difficulty_analysis": dict,
        "textbook_references": list,
        "video_references": list,
        "step_by_step_solution": dict,
        "formulas_principles": list,
        "real_world_applications": dict
    },
    "nested": {
        "answer_validation": {
            "required_fields": ["correct_answer", "is_correct", "confidence", "reasoning"],
            "field_types": {
                "correct_answer": str,
                "is_correct": bool,
                "confidence": float,
                "confidence_type": str,
                "reasoning": str
            }
        },
        "explanation": {
            "required_fields": ["question_nature", "step_by_step", "formulas_used"],
            "field_types": {
                "question_nature": str,
                "step_by_step": list,
                "formulas_used": list,
                "estimated_time_minutes": int
            }
        },
        "hierarchical_tags": {
            "required_fields": ["subject", "topic", "concepts"],
            "field_types": {
                "subject": dict,
                "topic": dict,
                "concepts": list
            }
        },
        "prerequisites": {
            "required_fields": ["essential"],
            "field_types": {
                "essential": list,
                "helpful": list,
                "dependency_tree": dict
            }
        },
        "difficulty_analysis": {
            "required_fields": ["overall", "score"],
            "field_types": {
                "overall": str,
                "score": int,
                "complexity_breakdown": dict,
                "difficulty_factors": list
            },
            "valid_values": {
                "overall": ["Very Easy", "Easy", "Medium", "Hard", "Very Hard"]
            },
            "value_ranges": {
                "score": (1, 10)
            }
        },
        "step_by_step_solution": {
            "required_fields": [],
            "field_types": {
                "approach_type": str,
                "total_steps": int,
                "solution_path": str,
                "key_insights": list
            }
        },
        "real_world_applications": {
            "required_fields": [],
            "field_types": {
                "industry_examples": list,
                "specific_systems": list,
                "practical_relevance": str
            }
        }
    },
    "array_schemas": {
        "hierarchical_tags.concepts": {
            "required_fields": ["name", "importance"],
            "field_types": {
                "name": str,
                "importance": str,
                "consensus": str
            }
        },
        "textbook_references": {
            "required_fields": ["book", "author", "relevance_score", "source"],
            "field_types": {
                "source_type": str,
                "book": str,
                "author": str,
                "relevance_score": float,
                "source": str
            }
        },
        "video_references": {
            "required_fields": ["professor", "topic_covered", "relevance_score", "source"],
            "field_types": {
                "source_type": str,
                "professor": str,
                "topic_covered": str,
                "relevance_score": float,
                "source": str
            }
        },
        "formulas_principles": {
            "required_fields": ["formula", "name", "relevance"],
            "field_types": {
                "formula": str,
                "name": str,
                "relevance": str
            }
        }
    }
}

TIER_2_SCHEMA = {
    "required_fields": [
        "common_mistakes",
        "mnemonics_memory_aids",
        "flashcards",
        "real_world_context",
        "exam_strategy"
    ],
    "field_types": {
        "common_mistakes": list,
        "mnemonics_memory_aids": list,
        "flashcards": list,
        "real_world_context": list,
        "exam_strategy": dict
    },
    "nested": {
        "exam_strategy": {
            "required_fields": ["priority", "triage_tip"],
            "field_types": {
                "priority": str,
                "triage_tip": str,
                "guessing_heuristic": str,
                "time_management": str
            },
            "valid_values": {
                "priority": ["Must Attempt", "Attempt If Time", "Skip If Unsure"]
            }
        }
    },
    "array_schemas": {
        "common_mistakes": {
            "required_fields": ["mistake", "type", "severity", "how_to_avoid"],
            "field_types": {
                "mistake": str,
                "type": str,
                "severity": str,
                "how_to_avoid": str
            }
        },
        "mnemonics_memory_aids": {
            "required_fields": ["mnemonic", "concept"],
            "field_types": {
                "mnemonic": str,
                "concept": str
            }
        },
        "flashcards": {
            "required_fields": ["card_type", "front", "back"],
            "field_types": {
                "card_type": str,
                "front": str,
                "back": str
            }
        },
        "real_world_context": {
            "required_fields": ["application", "industry_example"],
            "field_types": {
                "application": str,
                "industry_example": str
            }
        }
    }
}

TIER_3_SCHEMA = {
    "required_fields": [
        "search_keywords",
        "alternative_methods",
        "connections_to_other_subjects",
        "deeper_dive_topics"
    ],
    "field_types": {
        "search_keywords": list,
        "alternative_methods": list,
        "connections_to_other_subjects": dict,
        "deeper_dive_topics": list
    },
    "array_schemas": {
        "alternative_methods": {
            "required_fields": ["name", "description", "pros_cons"],
            "field_types": {
                "name": str,
                "description": str,
                "pros_cons": str
            }
        }
    }
}

TIER_4_SCHEMA = {
    "required_fields": [
        "question_metadata",
        "syllabus_mapping",
        "rag_quality",
        "model_meta",
        "future_questions_potential"
    ],
    "field_types": {
        "question_metadata": dict,
        "syllabus_mapping": dict,
        "rag_quality": dict,
        "model_meta": dict,
        "future_questions_potential": list
    },
    "nested": {
        "question_metadata": {
            "required_fields": ["id", "year", "marks"],
            "field_types": {
                "id": str,
                "year": int,
                "marks": float
            }
        },
        "syllabus_mapping": {
            "required_fields": ["gate_section"],
            "field_types": {
                "gate_section": str
            }
        },
        "rag_quality": {
            "required_fields": ["relevance_score", "notes"],
            "field_types": {
                "relevance_score": float,
                "notes": str,
                "sources_distribution": dict
            }
        },
        "model_meta": {
            "required_fields": ["timestamp", "version"],
            "field_types": {
                "timestamp": str,
                "version": str
            }
        }
    }
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_complete_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate complete JSON against all tier schemas
    
    Args:
        data: Complete JSON response
    
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    # Validate root fields
    root_required = [
        "question_id",
        "exam_name",
        "subject",
        "year",
        "question_text",
        "question_type"
    ]
    
    for field in root_required:
        if field not in data:
            errors.append(f"Missing root field: {field}")
    
    # Validate tier_0
    if "tier_0_classification" in data:
        tier_errors = validate_tier_with_schema(
            data["tier_0_classification"],
            TIER_0_SCHEMA,
            "tier_0_classification"
        )
        errors.extend(tier_errors)
    else:
        errors.append("Missing tier_0_classification")
    
    # Validate tier_1
    if "tier_1_core_research" in data:
        tier_errors = validate_tier_with_schema(
            data["tier_1_core_research"],
            TIER_1_SCHEMA,
            "tier_1_core_research"
        )
        errors.extend(tier_errors)
    else:
        errors.append("Missing tier_1_core_research")
    
    # Validate tier_2
    if "tier_2_student_learning" in data:
        tier_errors = validate_tier_with_schema(
            data["tier_2_student_learning"],
            TIER_2_SCHEMA,
            "tier_2_student_learning"
        )
        errors.extend(tier_errors)
    else:
        errors.append("Missing tier_2_student_learning")
    
    # Validate tier_3
    if "tier_3_enhanced_learning" in data:
        tier_errors = validate_tier_with_schema(
            data["tier_3_enhanced_learning"],
            TIER_3_SCHEMA,
            "tier_3_enhanced_learning"
        )
        errors.extend(tier_errors)
    else:
        errors.append("Missing tier_3_enhanced_learning")
    
    # Validate tier_4
    if "tier_4_metadata_and_future" in data:
        tier_errors = validate_tier_with_schema(
            data["tier_4_metadata_and_future"],
            TIER_4_SCHEMA,
            "tier_4_metadata_and_future"
        )
        errors.extend(tier_errors)
    else:
        errors.append("Missing tier_4_metadata_and_future")
    
    is_valid = len(errors) == 0
    
    if not is_valid:
        logger.warning(f"Schema validation found {len(errors)} errors")
    
    return is_valid, errors


def validate_tier_with_schema(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    tier_name: str
) -> List[str]:
    """
    Validate a single tier against its complete schema
    """
    errors = []
    
    # Check required fields
    required_fields = schema.get("required_fields", [])
    for field in required_fields:
        if field not in data:
            errors.append(f"{tier_name}: Missing required field '{field}'")
    
    # Check field types
    field_types = schema.get("field_types", {})
    for field, expected_type in field_types.items():
        if field in data:
            actual_value = data[field]
            
            if actual_value is None:
                errors.append(f"{tier_name}.{field}: Value is None")
                continue
            
            if not isinstance(actual_value, expected_type):
                errors.append(
                    f"{tier_name}.{field}: Expected {expected_type.__name__}, "
                    f"got {type(actual_value).__name__}"
                )
    
    # Check valid values
    valid_values = schema.get("valid_values", {})
    for field, allowed_values in valid_values.items():
        if field in data:
            value = data[field]
            if value not in allowed_values:
                errors.append(
                    f"{tier_name}.{field}: Invalid value '{value}'. "
                    f"Allowed: {allowed_values}"
                )
    
    # Check value ranges
    value_ranges = schema.get("value_ranges", {})
    for field, (min_val, max_val) in value_ranges.items():
        if field in data:
            value = data[field]
            if isinstance(value, (int, float)):
                if value < min_val or value > max_val:
                    errors.append(
                        f"{tier_name}.{field}: Value {value} outside range [{min_val}, {max_val}]"
                    )
    
    # Validate nested structures
    nested_schemas = schema.get("nested", {})
    array_schemas = schema.get("array_schemas", {})
    for field, nested_schema in nested_schemas.items():
        if field in data and data[field] is not None:
            nested_array_schemas = {
                k.replace(f"{field}.", ""): v 
                for k, v in array_schemas.items() 
                if k.startswith(f"{field}.")
            }
            nested_errors = validate_nested_structure(
                data[field],
                nested_schema,
                f"{tier_name}.{field}",
                nested_array_schemas=nested_array_schemas
            )
            errors.extend(nested_errors)
    
    # Validate array elements
    for field, array_schema in array_schemas.items():
        if field in data and isinstance(data[field], list):
            array_errors = validate_array_elements(
                data[field],
                array_schema,
                f"{tier_name}.{field}"
            )
            errors.extend(array_errors)
    
    return errors


def validate_nested_structure(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    context: str,
    nested_array_schemas: Dict[str, Any] = None
) -> List[str]:
    """
    Validate nested dictionary structure
    """
    errors = []
    
    if nested_array_schemas is None:
        nested_array_schemas = {}
    
    # Check required fields
    required_fields = schema.get("required_fields", [])
    for field in required_fields:
        if field not in data:
            errors.append(f"{context}: Missing required field '{field}'")
    
    # Check field types
    field_types = schema.get("field_types", {})
    for field, expected_type in field_types.items():
        if field in data:
            value = data[field]
            if value is not None and not isinstance(value, expected_type):
                errors.append(
                    f"{context}.{field}: Expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            
            if isinstance(value, list) and field in nested_array_schemas:
                array_errors = validate_array_elements(
                    value,
                    nested_array_schemas[field],
                    f"{context}.{field}"
                )
                errors.extend(array_errors)
    
    # Check valid values
    valid_values = schema.get("valid_values", {})
    for field, allowed_values in valid_values.items():
        if field in data:
            value = data[field]
            if value not in allowed_values:
                errors.append(
                    f"{context}.{field}: Invalid value '{value}'. Allowed: {allowed_values}"
                )
    
    # Check value ranges
    value_ranges = schema.get("value_ranges", {})
    for field, (min_val, max_val) in value_ranges.items():
        if field in data:
            value = data[field]
            if isinstance(value, (int, float)):
                if value < min_val or value > max_val:
                    errors.append(
                        f"{context}.{field}: Value {value} outside range [{min_val}, {max_val}]"
                    )
    
    return errors


def validate_array_elements(
    array: List[Any],
    schema: Dict[str, Any],
    context: str
) -> List[str]:
    """
    Validate elements in an array
    """
    errors = []
    
    required_fields = schema.get("required_fields", [])
    field_types = schema.get("field_types", {})
    
    for i, item in enumerate(array):
        if not isinstance(item, dict):
            errors.append(f"{context}[{i}]: Expected dict, got {type(item).__name__}")
            continue
        
        for field in required_fields:
            if field not in item:
                errors.append(f"{context}[{i}]: Missing required field '{field}'")
        
        for field, expected_type in field_types.items():
            if field in item:
                value = item[field]
                if value is not None and not isinstance(value, expected_type):
                    errors.append(
                        f"{context}[{i}].{field}: Expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
    
    return errors


# ============================================================================
# AUTO-FIX FUNCTIONS
# ============================================================================

def auto_fix_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-fix common JSON formatting issues
    """
    import copy
    fixed_data = copy.deepcopy(data)
    
    if "tier_0_classification" in fixed_data:
        fixed_data["tier_0_classification"] = fix_tier_0(
            fixed_data["tier_0_classification"]
        )
    
    if "tier_1_core_research" in fixed_data:
        fixed_data["tier_1_core_research"] = fix_tier_1(
            fixed_data["tier_1_core_research"]
        )
    
    if "tier_2_student_learning" in fixed_data:
        fixed_data["tier_2_student_learning"] = fix_tier_2(
            fixed_data["tier_2_student_learning"]
        )
    
    if "tier_3_enhanced_learning" in fixed_data:
        fixed_data["tier_3_enhanced_learning"] = fix_tier_3(
            fixed_data["tier_3_enhanced_learning"]
        )
    
    if "tier_4_metadata_and_future" in fixed_data:
        fixed_data["tier_4_metadata_and_future"] = fix_tier_4(
            fixed_data["tier_4_metadata_and_future"]
        )
    
    return fixed_data


def fix_tier_0(tier0: Dict[str, Any]) -> Dict[str, Any]:
    """Fix tier_0_classification"""
    if "difficulty_score" in tier0:
        tier0["difficulty_score"] = safe_int(tier0["difficulty_score"], default=5)
    
    if "use_gpt51" in tier0:
        tier0["use_gpt51"] = safe_bool(tier0["use_gpt51"], default=False)
    
    if "classification_confidence" in tier0:
        tier0["classification_confidence"] = safe_float(
            tier0["classification_confidence"], default=0.75
        )
    
    if "complexity_flags" not in tier0 or tier0["complexity_flags"] is None:
        tier0["complexity_flags"] = {}
    
    flag_fields = [
        "requires_derivation",
        "multi_concept_integration",
        "ambiguous_wording",
        "image_interpretation_complex",
        "edge_case_scenario",
        "multi_step_reasoning",
        "approximation_needed"
    ]
    
    for field in flag_fields:
        if field not in tier0["complexity_flags"]:
            tier0["complexity_flags"][field] = False
        else:
            tier0["complexity_flags"][field] = safe_bool(
                tier0["complexity_flags"][field], default=False
            )
    
    if "weight_strategy" not in tier0:
        tier0["weight_strategy"] = "BALANCED"
    
    return tier0


def fix_tier_1(tier1: Dict[str, Any]) -> Dict[str, Any]:
    """Fix tier_1_core_research"""
    
    if "answer_validation" not in tier1 or tier1["answer_validation"] is None:
        tier1["answer_validation"] = {}
    av = tier1["answer_validation"]
    if "correct_answer" not in av:
        av["correct_answer"] = ""
    if "is_correct" not in av:
        av["is_correct"] = True
    if "confidence" not in av:
        av["confidence"] = 0.8
    else:
        av["confidence"] = safe_float(av["confidence"], default=0.8)
    if "reasoning" not in av:
        av["reasoning"] = ""
    
    if "explanation" not in tier1 or tier1["explanation"] is None:
        tier1["explanation"] = {}
    exp = tier1["explanation"]
    if "question_nature" not in exp:
        exp["question_nature"] = "Calculation"
    if "step_by_step" not in exp or exp["step_by_step"] is None:
        exp["step_by_step"] = []
    if "formulas_used" not in exp or exp["formulas_used"] is None:
        exp["formulas_used"] = []
    
    if "hierarchical_tags" not in tier1 or tier1["hierarchical_tags"] is None:
        tier1["hierarchical_tags"] = {}
    tags = tier1["hierarchical_tags"]
    for tag_type in ["subject", "topic"]:
        if tag_type not in tags:
            tags[tag_type] = {"name": "", "confidence": 0.8}
    if "concepts" not in tags or tags["concepts"] is None:
        tags["concepts"] = []
    
    if "prerequisites" not in tier1 or tier1["prerequisites"] is None:
        tier1["prerequisites"] = {}
    prereq = tier1["prerequisites"]
    if "essential" not in prereq or prereq["essential"] is None:
        prereq["essential"] = []
    if "helpful" not in prereq or prereq["helpful"] is None:
        prereq["helpful"] = []
    if "dependency_tree" not in prereq or prereq["dependency_tree"] is None:
        prereq["dependency_tree"] = {}
    
    if "difficulty_analysis" not in tier1 or tier1["difficulty_analysis"] is None:
        tier1["difficulty_analysis"] = {}
    diff = tier1["difficulty_analysis"]
    if "overall" not in diff:
        diff["overall"] = "Medium"
    if "score" not in diff:
        diff["score"] = 5
    else:
        diff["score"] = safe_int(diff["score"], default=5)
    if "complexity_breakdown" not in diff:
        diff["complexity_breakdown"] = {}
    if "difficulty_factors" not in diff or diff["difficulty_factors"] is None:
        diff["difficulty_factors"] = []
    
    for list_field in ["textbook_references", "video_references", "formulas_principles"]:
        if list_field not in tier1 or tier1[list_field] is None:
            tier1[list_field] = []
    
    if "step_by_step_solution" not in tier1 or tier1["step_by_step_solution"] is None:
        tier1["step_by_step_solution"] = {}
    sbs = tier1["step_by_step_solution"]
    if "key_insights" not in sbs or sbs["key_insights"] is None:
        sbs["key_insights"] = []
    
    if "real_world_applications" not in tier1 or tier1["real_world_applications"] is None:
        tier1["real_world_applications"] = {}
    rwa = tier1["real_world_applications"]
    if "industry_examples" not in rwa or rwa["industry_examples"] is None:
        rwa["industry_examples"] = []
    if "specific_systems" not in rwa or rwa["specific_systems"] is None:
        rwa["specific_systems"] = []
    
    return tier1


def fix_tier_2(tier2: Dict[str, Any]) -> Dict[str, Any]:
    """Fix tier_2_student_learning"""
    
    list_fields = [
        "common_mistakes",
        "mnemonics_memory_aids",
        "flashcards",
        "real_world_context"
    ]
    for field in list_fields:
        if field not in tier2 or tier2[field] is None:
            tier2[field] = []
    
    if "exam_strategy" not in tier2 or tier2["exam_strategy"] is None:
        tier2["exam_strategy"] = {}
    es = tier2["exam_strategy"]
    if "priority" not in es:
        es["priority"] = "Must Attempt"
    if "triage_tip" not in es:
        es["triage_tip"] = ""
    
    return tier2


def fix_tier_3(tier3: Dict[str, Any]) -> Dict[str, Any]:
    """Fix tier_3_enhanced_learning"""
    
    list_fields = [
        "search_keywords",
        "alternative_methods",
        "deeper_dive_topics"
    ]
    for field in list_fields:
        if field not in tier3 or tier3[field] is None:
            tier3[field] = []
    
    if "connections_to_other_subjects" not in tier3 or tier3["connections_to_other_subjects"] is None:
        tier3["connections_to_other_subjects"] = {}
    
    return tier3


def fix_tier_4(tier4: Dict[str, Any]) -> Dict[str, Any]:
    """Fix tier_4_metadata_and_future"""
    
    if "question_metadata" not in tier4 or tier4["question_metadata"] is None:
        tier4["question_metadata"] = {}
    qm = tier4["question_metadata"]
    if "id" not in qm:
        qm["id"] = ""
    if "year" not in qm:
        qm["year"] = 2024
    else:
        qm["year"] = safe_int(qm["year"], default=2024)
    if "marks" not in qm:
        qm["marks"] = 1.0
    else:
        qm["marks"] = safe_float(qm["marks"], default=1.0)
    
    if "syllabus_mapping" not in tier4 or tier4["syllabus_mapping"] is None:
        tier4["syllabus_mapping"] = {}
    sm = tier4["syllabus_mapping"]
    if "gate_section" not in sm:
        sm["gate_section"] = ""
    
    if "rag_quality" not in tier4 or tier4["rag_quality"] is None:
        tier4["rag_quality"] = {}
    rq = tier4["rag_quality"]
    if "relevance_score" not in rq:
        rq["relevance_score"] = 0.5
    else:
        rq["relevance_score"] = safe_float(rq["relevance_score"], default=0.5)
    if "notes" not in rq:
        rq["notes"] = ""
    if "sources_distribution" not in rq or rq["sources_distribution"] is None:
        rq["sources_distribution"] = {}
    
    if "model_meta" not in tier4 or tier4["model_meta"] is None:
        tier4["model_meta"] = {}
    mm = tier4["model_meta"]
    if "timestamp" not in mm:
        mm["timestamp"] = ""
    if "version" not in mm:
        mm["version"] = "GATE_AE_SOTA_v1.0"
    
    if "future_questions_potential" not in tier4 or tier4["future_questions_potential"] is None:
        tier4["future_questions_potential"] = []
    
    return tier4


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    if isinstance(value, int):
        return value
    
    if isinstance(value, float):
        return int(value)
    
    if isinstance(value, str):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default
    
    return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    if isinstance(value, float):
        return value
    
    if isinstance(value, int):
        return float(value)
    
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    return default


def safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert value to bool"""
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        return value.lower() in ["true", "1", "yes", "y"]
    
    if isinstance(value, int):
        return value != 0
    
    return default


def sanitize_json_string(json_string: str) -> str:
    """
    Clean up JSON string before parsing
    
    Fixes:
    - Remove markdown code blocks (```json ... ```)
    - Strip whitespace
    - Remove BOM if present
    """
    json_string = json_string.strip()
    
    # Remove markdown code blocks
    if json_string.startswith("```json"):
        json_string = json_string.replace("```json", "", 1)
    if json_string.startswith("```"):
        json_string = json_string.replace("```", "", 1)
    if json_string.endswith("```"):
        json_string = json_string.rsplit("```", 1)[0]
    
    # Strip whitespace
    json_string = json_string.strip()
    
    # Remove BOM if present
    if json_string.startswith('\ufeff'):
        json_string = json_string[1:]
    
    return json_string


def parse_json_safely(json_string: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON string with error handling
    
    Args:
        json_string: JSON string to parse
    
    Returns:
        dict: Parsed JSON or None if parsing failed
    """
    clean_string = sanitize_json_string(json_string)
    
    try:
        data = json.loads(clean_string)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Failed at position {e.pos}: {clean_string[max(0, e.pos-50):e.pos+50]}")
        return None


def get_validation_summary(errors: List[str]) -> Dict[str, Any]:
    """
    Generate summary of validation errors
    
    Args:
        errors: List of error messages
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        "total_errors": len(errors),
        "by_tier": {},
        "error_types": {}
    }
    
    for error in errors:
        # Extract tier name
        if "tier_" in error:
            tier = error.split(".")[0].split(":")[0]
            summary["by_tier"][tier] = summary["by_tier"].get(tier, 0) + 1
        
        # Categorize error type
        if "Missing" in error:
            summary["error_types"]["missing_field"] = summary["error_types"].get("missing_field", 0) + 1
        elif "Expected" in error:
            summary["error_types"]["type_mismatch"] = summary["error_types"].get("type_mismatch", 0) + 1
        elif "Invalid value" in error:
            summary["error_types"]["invalid_value"] = summary["error_types"].get("invalid_value", 0) + 1
        elif "outside range" in error:
            summary["error_types"]["out_of_range"] = summary["error_types"].get("out_of_range", 0) + 1
        else:
            summary["error_types"]["other"] = summary["error_types"].get("other", 0) + 1
    
    return summary