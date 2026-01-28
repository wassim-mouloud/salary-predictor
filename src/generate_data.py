"""
Generate synthetic salary dataset for the Salary Predictor project.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_salary_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic salary dataset.

    Features:
    - years_experience: Years of professional experience (0-30)
    - education_level: Encoded education (1=High School, 2=Bachelor's, 3=Master's, 4=PhD)
    - age: Age of the employee (22-60)
    - job_role: Encoded job role (1=Junior, 2=Mid, 3=Senior, 4=Lead, 5=Manager)

    Target:
    - salary: Annual salary in USD
    """
    np.random.seed(random_state)

    # Generate features
    years_experience = np.random.uniform(0, 30, n_samples)
    education_level = np.random.choice([1, 2, 3, 4], n_samples, p=[0.15, 0.45, 0.30, 0.10])
    age = np.clip(22 + years_experience + np.random.normal(0, 3, n_samples), 22, 60)
    job_role = np.clip(
        1 + (years_experience // 5).astype(int) + np.random.randint(-1, 2, n_samples),
        1, 5
    ).astype(int)

    base_salary = 35000
    salary = (
        base_salary
        + years_experience * 2500  # Experience bonus
        + education_level * 8000   # Education bonus
        + job_role * 12000         # Role bonus
        + np.random.normal(0, 5000, n_samples)  # Random variation
    )
    salary = np.clip(salary, 30000, 250000)

    df = pd.DataFrame({
        'years_experience': np.round(years_experience, 1),
        'education_level': education_level,
        'age': np.round(age, 0).astype(int),
        'job_role': job_role,
        'salary': np.round(salary, -2)  # Round to nearest 100
    })

    education_map = {1: 'High School', 2: "Bachelor's", 3: "Master's", 4: 'PhD'}
    role_map = {1: 'Junior', 2: 'Mid-Level', 3: 'Senior', 4: 'Lead', 5: 'Manager'}

    df['education_name'] = df['education_level'].map(education_map)
    df['role_name'] = df['job_role'].map(role_map)

    return df


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    df = generate_salary_data(n_samples=1000)
    df.to_csv(data_dir / "salary_data.csv", index=False)

    print(f"Dataset generated with {len(df)} samples")
    print(f"Saved to {data_dir / 'salary_data.csv'}")
    print("\nDataset Preview:")
    print(df.head(10))
    print("\nDataset Statistics:")
    print(df.describe())
