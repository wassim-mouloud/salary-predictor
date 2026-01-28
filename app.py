"""
Streamlit web application for Salary Prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import SalaryPredictor
from visualizations import (
    plot_salary_distribution,
    plot_experience_vs_salary,
    plot_salary_by_education,
    plot_salary_by_role,
    plot_correlation_matrix,
    plot_actual_vs_predicted,
    set_style
)

st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide"
)

EDUCATION_OPTIONS = {
    'High School': 1,
    "Bachelor's Degree": 2,
    "Master's Degree": 3,
    'PhD': 4
}

ROLE_OPTIONS = {
    'Junior': 1,
    'Mid-Level': 2,
    'Senior': 3,
    'Lead': 4,
    'Manager': 5
}


@st.cache_resource
def load_model():
    """Load the trained model."""
    model_path = Path(__file__).parent / "models" / "salary_model.joblib"
    predictor = SalaryPredictor()

    if model_path.exists():
        predictor.load(model_path)
    else:
        data_path = Path(__file__).parent / "data" / "salary_data.csv"
        if not data_path.exists():
            from generate_data import generate_salary_data
            df = generate_salary_data()
            data_path.parent.mkdir(exist_ok=True)
            df.to_csv(data_path, index=False)

        df = pd.read_csv(data_path)
        predictor.fit(df)
        model_path.parent.mkdir(exist_ok=True)
        predictor.save(model_path)

    return predictor


@st.cache_data
def load_data():
    """Load the dataset."""
    data_path = Path(__file__).parent / "data" / "salary_data.csv"
    if not data_path.exists():
        from generate_data import generate_salary_data
        df = generate_salary_data()
        data_path.parent.mkdir(exist_ok=True)
        df.to_csv(data_path, index=False)
        return df
    return pd.read_csv(data_path)


def main():
    st.title("💰 Salary Predictor")
    st.markdown("*A Machine Learning model to predict salaries based on experience, education, and job role.*")
    st.markdown("---")

    predictor = load_model()
    df = load_data()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🎯 Predict Salary", "📊 Data Exploration", "📈 Model Performance", "ℹ️ About"]
    )

    if page == "🎯 Predict Salary":
        predict_page(predictor)
    elif page == "📊 Data Exploration":
        exploration_page(df)
    elif page == "📈 Model Performance":
        performance_page(predictor, df)
    else:
        about_page()


def predict_page(predictor):
    """Salary prediction page."""
    st.header("Predict Your Salary")
    st.markdown("Enter your details below to get a salary prediction.")

    col1, col2 = st.columns(2)

    with col1:
        years_experience = st.slider(
            "Years of Experience",
            min_value=0.0,
            max_value=30.0,
            value=5.0,
            step=0.5,
            help="Total years of professional experience"
        )

        education = st.selectbox(
            "Education Level",
            options=list(EDUCATION_OPTIONS.keys()),
            index=1,
            help="Highest education level completed"
        )

    with col2:
        age = st.slider(
            "Age",
            min_value=22,
            max_value=60,
            value=28,
            help="Your current age"
        )

        job_role = st.selectbox(
            "Job Role",
            options=list(ROLE_OPTIONS.keys()),
            index=1,
            help="Current or target job role"
        )

    st.markdown("---")

    if st.button("🔮 Predict Salary", type="primary", use_container_width=True):
        education_level = EDUCATION_OPTIONS[education]
        role_level = ROLE_OPTIONS[job_role]

        prediction = predictor.predict(
            years_experience=years_experience,
            education_level=education_level,
            age=age,
            job_role=role_level
        )

        st.markdown("### Predicted Salary")
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin: 20px 0;
            ">
                <h1 style="color: white; margin: 0; font-size: 3em;">
                    ${prediction:,.0f}
                </h1>
                <p style="color: rgba(255,255,255,0.8); margin-top: 10px;">
                    Estimated Annual Salary
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Input Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Experience", f"{years_experience} years")
        col2.metric("Education", education)
        col3.metric("Age", f"{age} years")
        col4.metric("Role", job_role)


def exploration_page(df):
    """Data exploration page."""
    st.header("Data Exploration")

    set_style()

    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", f"{len(df):,}")
    col2.metric("Avg Salary", f"${df['salary'].mean():,.0f}")
    col3.metric("Min Salary", f"${df['salary'].min():,.0f}")
    col4.metric("Max Salary", f"${df['salary'].max():,.0f}")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Salary Distribution",
        "Experience vs Salary",
        "By Education",
        "By Role",
        "Correlations"
    ])

    with tab1:
        fig = plot_salary_distribution(df)
        st.pyplot(fig)

    with tab2:
        fig = plot_experience_vs_salary(df)
        st.pyplot(fig)

    with tab3:
        fig = plot_salary_by_education(df)
        st.pyplot(fig)

    with tab4:
        fig = plot_salary_by_role(df)
        st.pyplot(fig)

    with tab5:
        fig = plot_correlation_matrix(df)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Raw Data")
    if st.checkbox("Show raw data"):
        st.dataframe(df, use_container_width=True)


def performance_page(predictor, df):
    """Model performance page."""
    st.header("Model Performance")

    set_style()

    metrics = predictor.fit(df)

    st.subheader("Model Evaluation Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Training Set")
        st.metric("R² Score", f"{metrics['train']['r2']:.4f}")
        st.metric("RMSE", f"${metrics['train']['rmse']:,.0f}")
        st.metric("MAE", f"${metrics['train']['mae']:,.0f}")

    with col2:
        st.markdown("#### Test Set")
        st.metric("R² Score", f"{metrics['test']['r2']:.4f}")
        st.metric("RMSE", f"${metrics['test']['rmse']:,.0f}")
        st.metric("MAE", f"${metrics['test']['mae']:,.0f}")

    st.markdown("---")

    st.subheader("Feature Coefficients")
    coef_df = pd.DataFrame({
        'Feature': list(metrics['coefficients'].keys()),
        'Coefficient': list(metrics['coefficients'].values())
    }).sort_values('Coefficient', ascending=False)

    st.bar_chart(coef_df.set_index('Feature'))

    st.markdown("""
    **Interpretation:**
    - Positive coefficients increase salary prediction
    - The magnitude shows the relative importance of each feature
    - These are scaled coefficients (features are standardized)
    """)

    st.markdown("---")
    st.subheader("Actual vs Predicted")

    y_actual = df['salary'].values
    y_pred = predictor.predict_batch(df)
    fig = plot_actual_vs_predicted(y_actual, y_pred)
    st.pyplot(fig)


def about_page():
    """About page."""
    st.header("About This Project")

    st.markdown("""
    ### Salary Predictor

    This project demonstrates a **Linear Regression** machine learning model
    to predict salaries based on various factors.

    #### Features Used
    - **Years of Experience**: Professional work experience
    - **Education Level**: Highest degree completed
    - **Age**: Current age
    - **Job Role**: Position level in the organization

    #### Technology Stack
    - **Python**: Core programming language
    - **scikit-learn**: Machine learning model
    - **Pandas**: Data manipulation
    - **Matplotlib/Seaborn**: Data visualization
    - **Streamlit**: Web interface

    #### Model Details
    - Algorithm: Linear Regression
    - Features are standardized using StandardScaler
    - Train/Test split: 80/20

    #### Project Structure
    ```
    salary-predictor/
    ├── app.py              # Streamlit web app
    ├── data/               # Dataset storage
    ├── models/             # Trained models
    ├── src/
    │   ├── generate_data.py   # Data generation
    │   ├── model.py           # ML model
    │   └── visualizations.py  # Plotting functions
    ├── notebooks/          # Jupyter notebooks
    └── requirements.txt    # Dependencies
    ```

    ---
    *Built as a portfolio project demonstrating machine learning skills.*
    """)


if __name__ == "__main__":
    main()
