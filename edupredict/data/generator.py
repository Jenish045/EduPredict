"""
Synthetic student academic performance dataset generator.

Creates realistic, reproducible educational data for training and evaluating
regression (predicting final exam scores) and classification (predicting academic risk).
Deterministic generation supported via configurable random seed.
"""

import random
from typing import Tuple, List, Any, Optional
from edupredict.data.dataset import Dataset


def generate_student_data(
    n_samples: int = 500,
    random_seed: Optional[int] = 42,
    missing_rate: float = 0.02
) -> Dataset:
    """
    Generate synthetic educational dataset.

    Features:
    - study_hours: float, weekly study hours (4.0 to 40.0)
    - attendance: float, class attendance percentage (45.0 to 100.0)
    - past_score: float, past semester exam score (30.0 to 95.0)
    - assignments_completed: int, assignments turned in (0 to 10)
    - parental_education: str ('HighSchool', 'Bachelor', 'Master')
    - internet_access: str ('Yes', 'No')
    - tutoring: str ('Yes', 'No')
    
    Targets:
    - final_exam_score: float, continuous score [0.0, 100.0]
    - academic_risk: int, 1 if final_exam_score < 60.0 else 0
    """
    if random_seed is not None:
        rng = random.Random(random_seed)
    else:
        rng = random.Random()

    headers = [
        "student_id",
        "study_hours",
        "attendance",
        "past_score",
        "assignments_completed",
        "parental_education",
        "internet_access",
        "tutoring",
        "final_exam_score",
        "academic_risk"
    ]

    education_levels = ["HighSchool", "Bachelor", "Master"]
    binary_choices = ["No", "Yes"]

    rows: List[List[Any]] = []

    for i in range(1, n_samples + 1):
        sid = f"STU_{i:04d}"
        
        # Base independent features with realistic distributions
        study = round(rng.uniform(5.0, 35.0), 1)
        attendance = round(rng.uniform(50.0, 100.0), 1)
        past = round(rng.uniform(35.0, 95.0), 1)
        assignments = rng.randint(2, 10)
        
        parent_edu = rng.choice(education_levels)
        internet = rng.choices(binary_choices, weights=[0.2, 0.8])[0]
        tutoring = rng.choices(binary_choices, weights=[0.7, 0.3])[0]

        # Calculate academic performance based on classical factors + slight noise
        edu_bonus = 0.0 if parent_edu == "HighSchool" else (2.5 if parent_edu == "Bachelor" else 5.0)
        net_bonus = 3.0 if internet == "Yes" else 0.0
        tutor_bonus = 4.0 if tutoring == "Yes" else 0.0
        
        noise = rng.gauss(0.0, 3.5)
        
        raw_final = (
            0.30 * past +
            0.30 * attendance +
            0.75 * study +
            1.20 * assignments +
            edu_bonus +
            net_bonus +
            tutor_bonus -
            10.0 +
            noise
        )
        
        final_score = round(max(0.0, min(100.0, raw_final)), 1)
        academic_risk = 1 if final_score < 60.0 else 0

        # Inject missing values if requested (simulating real-world messy student records)
        row: List[Any] = [sid, study, attendance, past, assignments, parent_edu, internet, tutoring, final_score, academic_risk]
        
        if missing_rate > 0:
            # Randomly blank out some feature cells (not id or targets)
            for col_idx in range(1, 8):
                if rng.random() < missing_rate:
                    row[col_idx] = None

        rows.append(row)

    return Dataset(headers=headers, data=rows, target_col="final_exam_score")
