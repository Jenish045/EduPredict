"""
Metadata definitions, user-facing display labels, and input control configurations
for the UCI Student Performance Mathematics dataset (student-mat.csv).

Provides a single authoritative mapping for display labels, input types, valid
options, format functions, and concise explanations without altering internal
schemas or pipeline representations.
"""

from typing import Dict, Any, List


# -----------------------------------------------------------------------------
# 1. Single Authoritative Display Name Mapping
# -----------------------------------------------------------------------------
UCI_DISPLAY_LABELS: Dict[str, str] = {
    # 30 Non-grade predictor features
    "school": "School",
    "sex": "Sex",
    "age": "Age (years)",
    "address": "Residential Area",
    "famsize": "Family Size",
    "Pstatus": "Parents' Cohabitation Status",
    "Medu": "Mother's Education Level",
    "Fedu": "Father's Education Level",
    "Mjob": "Mother's Occupation",
    "Fjob": "Father's Occupation",
    "reason": "Reason for Choosing School",
    "guardian": "Student's Guardian",
    "traveltime": "Travel Time to School",
    "studytime": "Weekly Study Time",
    "failures": "Previous Class Failures",
    "schoolsup": "Extra Educational Support",
    "famsup": "Family Educational Support",
    "paid": "Extra Paid Classes",
    "activities": "Extracurricular Activities",
    "nursery": "Attended Nursery School",
    "higher": "Wants Higher Education",
    "internet": "Internet Access at Home",
    "romantic": "In a Romantic Relationship",
    "famrel": "Family Relationship Quality",
    "freetime": "Free Time After School",
    "goout": "Going Out with Friends",
    "Dalc": "Workday Alcohol Consumption",
    "Walc": "Weekend Alcohol Consumption",
    "health": "Self-Rated Health",
    "absences": "School Absences",
    # Targets and course outcomes
    "G1": "First Period Grade (G1)",
    "G2": "Second Period Grade (G2)",
    "G3": "Final Mathematics Grade (G3)",
    "academic_risk": "Academic Risk",
}


def get_column_display_name(col: str) -> str:
    """Return friendly display label from UCI_DISPLAY_LABELS, or formatted title."""
    return UCI_DISPLAY_LABELS.get(col, col.replace("_", " ").title())


# -----------------------------------------------------------------------------
# 2. Concise Explanations for Coded Categories (UCI Documentation)
# -----------------------------------------------------------------------------
EDUCATION_LEVEL_LABELS: Dict[int, str] = {
    0: "0 — None",
    1: "1 — Primary education (4th grade)",
    2: "2 — 5th–9th grade",
    3: "3 — Secondary education",
    4: "4 — Higher education",
}

TRAVEL_TIME_LABELS: Dict[int, str] = {
    1: "1 — Less than 15 minutes",
    2: "2 — 15–30 minutes",
    3: "3 — 30 minutes to 1 hour",
    4: "4 — More than 1 hour",
}

STUDY_TIME_LABELS: Dict[int, str] = {
    1: "1 — Less than 2 hours/week",
    2: "2 — 2–5 hours/week",
    3: "3 — 5–10 hours/week",
    4: "4 — More than 10 hours/week",
}

ADDRESS_LABELS: Dict[str, str] = {
    "U": "U (Urban)",
    "R": "R (Rural)",
}

FAMILY_SIZE_LABELS: Dict[str, str] = {
    "LE3": "LE3 (3 or fewer)",
    "GT3": "GT3 (More than 3)",
}

COHABITATION_LABELS: Dict[str, str] = {
    "T": "T (Together)",
    "A": "A (Apart)",
}

SCALE_1_TO_5_HELP: str = "1 is the lowest level and 5 is the highest."


# -----------------------------------------------------------------------------
# 3. Interactive Input Control Configuration (Streamlit Forms)
# -----------------------------------------------------------------------------
UCI_FEATURE_CONFIG: Dict[str, Dict[str, Any]] = {
    "school": {
        "label": UCI_DISPLAY_LABELS["school"],
        "control": "select",
        "options": ["GP", "MS"],
        "default": "GP",
        "format_func": lambda x: str(x),
        "help": "Student's school: GP (Gabriel Pereira) or MS (Mousinho da Silveira).",
    },
    "sex": {
        "label": UCI_DISPLAY_LABELS["sex"],
        "control": "select",
        "options": ["F", "M"],
        "default": "F",
        "format_func": lambda x: str(x),
        "help": "Student's sex: F (female) or M (male).",
    },
    "age": {
        "label": UCI_DISPLAY_LABELS["age"],
        "control": "slider",
        "min": 15,
        "max": 22,
        "default": 17,
        "step": 1,
        "help": "Student's age in years (from 15 to 22).",
    },
    "address": {
        "label": UCI_DISPLAY_LABELS["address"],
        "control": "select",
        "options": ["U", "R"],
        "default": "U",
        "format_func": lambda x: ADDRESS_LABELS.get(x, str(x)),
        "help": "Student's home address type: U (Urban) or R (Rural).",
    },
    "famsize": {
        "label": UCI_DISPLAY_LABELS["famsize"],
        "control": "select",
        "options": ["LE3", "GT3"],
        "default": "GT3",
        "format_func": lambda x: FAMILY_SIZE_LABELS.get(x, str(x)),
        "help": "Family size: LE3 (3 or fewer) or GT3 (More than 3).",
    },
    "Pstatus": {
        "label": UCI_DISPLAY_LABELS["Pstatus"],
        "control": "select",
        "options": ["T", "A"],
        "default": "T",
        "format_func": lambda x: COHABITATION_LABELS.get(x, str(x)),
        "help": "Parent's cohabitation status: T (Living together) or A (Living apart).",
    },
    "Medu": {
        "label": UCI_DISPLAY_LABELS["Medu"],
        "control": "select",
        "options": [0, 1, 2, 3, 4],
        "default": 4,
        "format_func": lambda x: EDUCATION_LEVEL_LABELS.get(x, str(x)),
        "help": "Mother's education level (0: None to 4: Higher education).",
    },
    "Fedu": {
        "label": UCI_DISPLAY_LABELS["Fedu"],
        "control": "select",
        "options": [0, 1, 2, 3, 4],
        "default": 2,
        "format_func": lambda x: EDUCATION_LEVEL_LABELS.get(x, str(x)),
        "help": "Father's education level (0: None to 4: Higher education).",
    },
    "Mjob": {
        "label": UCI_DISPLAY_LABELS["Mjob"],
        "control": "select",
        "options": ["teacher", "health", "services", "at_home", "other"],
        "default": "other",
        "format_func": lambda x: str(x),
        "help": "Mother's occupation category.",
    },
    "Fjob": {
        "label": UCI_DISPLAY_LABELS["Fjob"],
        "control": "select",
        "options": ["teacher", "health", "services", "at_home", "other"],
        "default": "other",
        "format_func": lambda x: str(x),
        "help": "Father's occupation category.",
    },
    "reason": {
        "label": UCI_DISPLAY_LABELS["reason"],
        "control": "select",
        "options": ["home", "reputation", "course", "other"],
        "default": "course",
        "format_func": lambda x: str(x),
        "help": "Reason for choosing this school: close to home, reputation, course preference, or other.",
    },
    "guardian": {
        "label": UCI_DISPLAY_LABELS["guardian"],
        "control": "select",
        "options": ["mother", "father", "other"],
        "default": "mother",
        "format_func": lambda x: str(x),
        "help": "Student's primary legal guardian.",
    },
    "traveltime": {
        "label": UCI_DISPLAY_LABELS["traveltime"],
        "control": "select",
        "options": [1, 2, 3, 4],
        "default": 1,
        "format_func": lambda x: TRAVEL_TIME_LABELS.get(x, str(x)),
        "help": "Home to school travel time (1: <15 min to 4: >1 hour).",
    },
    "studytime": {
        "label": UCI_DISPLAY_LABELS["studytime"],
        "control": "select",
        "options": [1, 2, 3, 4],
        "default": 2,
        "format_func": lambda x: STUDY_TIME_LABELS.get(x, str(x)),
        "help": "Weekly study time (1: <2 hours to 4: >10 hours).",
    },
    "failures": {
        "label": UCI_DISPLAY_LABELS["failures"],
        "control": "select",
        "options": [0, 1, 2, 3],
        "default": 0,
        "format_func": lambda x: str(x),
        "help": "Number of past class failures (0 to 3).",
    },
    "schoolsup": {
        "label": UCI_DISPLAY_LABELS["schoolsup"],
        "control": "select",
        "options": ["yes", "no"],
        "default": "no",
        "format_func": lambda x: str(x),
        "help": "Extra educational school support.",
    },
    "famsup": {
        "label": UCI_DISPLAY_LABELS["famsup"],
        "control": "select",
        "options": ["yes", "no"],
        "default": "yes",
        "format_func": lambda x: str(x),
        "help": "Family-provided educational support.",
    },
    "paid": {
        "label": UCI_DISPLAY_LABELS["paid"],
        "control": "select",
        "options": ["yes", "no"],
        "default": "no",
        "format_func": lambda x: str(x),
        "help": "Extra paid classes within the course subject.",
    },
    "activities": {
        "label": UCI_DISPLAY_LABELS["activities"],
        "control": "select",
        "options": ["yes", "no"],
        "default": "yes",
        "format_func": lambda x: str(x),
        "help": "Participation in extracurricular activities.",
    },
    "nursery": {
        "label": UCI_DISPLAY_LABELS["nursery"],
        "control": "select",
        "options": ["yes", "no"],
        "default": "yes",
        "format_func": lambda x: str(x),
        "help": "Attended nursery school as a child.",
    },
    "higher": {
        "label": UCI_DISPLAY_LABELS["higher"],
        "control": "select",
        "options": ["yes", "no"],
        "default": "yes",
        "format_func": lambda x: str(x),
        "help": "Desire to pursue higher education.",
    },
    "internet": {
        "label": UCI_DISPLAY_LABELS["internet"],
        "control": "select",
        "options": ["yes", "no"],
        "default": "yes",
        "format_func": lambda x: str(x),
        "help": "Internet access at home.",
    },
    "romantic": {
        "label": UCI_DISPLAY_LABELS["romantic"],
        "control": "select",
        "options": ["yes", "no"],
        "default": "no",
        "format_func": lambda x: str(x),
        "help": "Currently in a romantic relationship.",
    },
    "famrel": {
        "label": UCI_DISPLAY_LABELS["famrel"],
        "control": "select",
        "options": [1, 2, 3, 4, 5],
        "default": 4,
        "format_func": lambda x: str(x),
        "help": SCALE_1_TO_5_HELP,
    },
    "freetime": {
        "label": UCI_DISPLAY_LABELS["freetime"],
        "control": "select",
        "options": [1, 2, 3, 4, 5],
        "default": 3,
        "format_func": lambda x: str(x),
        "help": SCALE_1_TO_5_HELP,
    },
    "goout": {
        "label": UCI_DISPLAY_LABELS["goout"],
        "control": "select",
        "options": [1, 2, 3, 4, 5],
        "default": 3,
        "format_func": lambda x: str(x),
        "help": SCALE_1_TO_5_HELP,
    },
    "Dalc": {
        "label": UCI_DISPLAY_LABELS["Dalc"],
        "control": "select",
        "options": [1, 2, 3, 4, 5],
        "default": 1,
        "format_func": lambda x: str(x),
        "help": SCALE_1_TO_5_HELP,
    },
    "Walc": {
        "label": UCI_DISPLAY_LABELS["Walc"],
        "control": "select",
        "options": [1, 2, 3, 4, 5],
        "default": 1,
        "format_func": lambda x: str(x),
        "help": SCALE_1_TO_5_HELP,
    },
    "health": {
        "label": UCI_DISPLAY_LABELS["health"],
        "control": "select",
        "options": [1, 2, 3, 4, 5],
        "default": 5,
        "format_func": lambda x: str(x),
        "help": SCALE_1_TO_5_HELP,
    },
    "absences": {
        "label": UCI_DISPLAY_LABELS["absences"],
        "control": "number",
        "min": 0,
        "max": 150,
        "default": 0,
        "step": 1,
        "help": "Total number of school absences (dataset observed maximum: 75).",
    },
}


# -----------------------------------------------------------------------------
# 4. Helpers for Previews & Backwards Compatibility
# -----------------------------------------------------------------------------
def get_uci_feature_config() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the UCI feature configuration dictionary."""
    return dict(UCI_FEATURE_CONFIG)


# Target outcomes dictionary (alias to relevant keys in UCI_DISPLAY_LABELS)
DATASET_OUTCOME_LABELS: Dict[str, str] = {
    k: UCI_DISPLAY_LABELS[k] for k in ("G1", "G2", "G3", "academic_risk")
}


def format_preview_records(headers: List[str], rows: List[List[Any]]) -> List[Dict[str, Any]]:
    """
    Create a display-only copy of dataset records with friendly column headers.
    Preserves exact row order, values, and original column ordering.
    """
    display_headers = [get_column_display_name(h) for h in headers]
    return [
        {display_headers[i]: row[i] for i in range(len(headers))}
        for row in rows
    ]
