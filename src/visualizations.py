"""
Data visualizations for the Salary Predictor project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def set_style():
    """Set consistent plot styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")


def plot_salary_distribution(df: pd.DataFrame, save_path: str = None):
    """Plot the distribution of salaries."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(df['salary'], bins=30, kde=True, ax=ax, color='steelblue')
    ax.set_xlabel('Salary ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Salaries', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_experience_vs_salary(df: pd.DataFrame, save_path: str = None):
    """Plot years of experience vs salary with regression line."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.regplot(
        data=df, x='years_experience', y='salary',
        scatter_kws={'alpha': 0.5, 's': 30},
        line_kws={'color': 'red', 'linewidth': 2},
        ax=ax
    )
    ax.set_xlabel('Years of Experience', fontsize=12)
    ax.set_ylabel('Salary ($)', fontsize=12)
    ax.set_title('Years of Experience vs Salary', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_salary_by_education(df: pd.DataFrame, save_path: str = None):
    """Plot salary distribution by education level."""
    fig, ax = plt.subplots(figsize=(10, 6))

    order = ['High School', "Bachelor's", "Master's", 'PhD']
    sns.boxplot(data=df, x='education_name', y='salary', order=order, ax=ax)
    ax.set_xlabel('Education Level', fontsize=12)
    ax.set_ylabel('Salary ($)', fontsize=12)
    ax.set_title('Salary by Education Level', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_salary_by_role(df: pd.DataFrame, save_path: str = None):
    """Plot salary distribution by job role."""
    fig, ax = plt.subplots(figsize=(10, 6))

    order = ['Junior', 'Mid-Level', 'Senior', 'Lead', 'Manager']
    sns.boxplot(data=df, x='role_name', y='salary', order=order, ax=ax)
    ax.set_xlabel('Job Role', fontsize=12)
    ax.set_ylabel('Salary ($)', fontsize=12)
    ax.set_title('Salary by Job Role', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    """Plot correlation matrix of numeric features."""
    fig, ax = plt.subplots(figsize=(8, 6))

    numeric_cols = ['years_experience', 'education_level', 'age', 'job_role', 'salary']
    corr_matrix = df[numeric_cols].corr()

    sns.heatmap(
        corr_matrix, annot=True, cmap='RdBu_r', center=0,
        fmt='.2f', square=True, ax=ax,
        cbar_kws={'shrink': 0.8}
    )
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_actual_vs_predicted(y_actual: np.ndarray, y_predicted: np.ndarray,
                             save_path: str = None):
    """Plot actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(y_actual, y_predicted, alpha=0.5, s=30)

    # Perfect prediction line
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Salary ($)', fontsize=12)
    ax.set_ylabel('Predicted Salary ($)', fontsize=12)
    ax.set_title('Actual vs Predicted Salaries', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_residuals(y_actual: np.ndarray, y_predicted: np.ndarray, save_path: str = None):
    """Plot residuals distribution."""
    residuals = y_actual - y_predicted

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_predicted, residuals, alpha=0.5, s=30)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Salary ($)', fontsize=12)
    axes[0].set_ylabel('Residuals ($)', fontsize=12)
    axes[0].set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Residuals distribution
    sns.histplot(residuals, bins=30, kde=True, ax=axes[1], color='steelblue')
    axes[1].set_xlabel('Residuals ($)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def generate_all_plots(df: pd.DataFrame, output_dir: str = None):
    """Generate and save all plots."""
    set_style()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        plot_salary_distribution(df, output_path / "salary_distribution.png")
        plot_experience_vs_salary(df, output_path / "experience_vs_salary.png")
        plot_salary_by_education(df, output_path / "salary_by_education.png")
        plot_salary_by_role(df, output_path / "salary_by_role.png")
        plot_correlation_matrix(df, output_path / "correlation_matrix.png")

        print(f"All plots saved to {output_path}")
    else:
        plot_salary_distribution(df)
        plot_experience_vs_salary(df)
        plot_salary_by_education(df)
        plot_salary_by_role(df)
        plot_correlation_matrix(df)
        plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    df = pd.read_csv(project_root / "data" / "salary_data.csv")
    generate_all_plots(df, project_root / "outputs")
