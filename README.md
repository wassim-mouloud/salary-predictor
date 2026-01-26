# Salary Predictor

A machine learning project that predicts salaries using **Linear Regression** based on years of experience, education level, age, and job role.

## Features

- Linear Regression model with scikit-learn
- Interactive Streamlit web interface
- Data visualizations with Matplotlib and Seaborn
- Model evaluation metrics (R², RMSE, MAE)
- Synthetic dataset generation

## Demo

![Salary Predictor Demo](outputs/demo.png)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/salary-predictor.git
cd salary-predictor
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Dataset
```bash
python src/generate_data.py
```

### Train the Model
```bash
python src/model.py
```

### Run the Web App
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## Project Structure

```
salary-predictor/
├── app.py                 # Streamlit web application
├── data/
│   └── salary_data.csv    # Generated dataset
├── models/
│   └── salary_model.joblib # Trained model
├── src/
│   ├── generate_data.py   # Dataset generation
│   ├── model.py           # Linear Regression model
│   └── visualizations.py  # Plotting functions
├── notebooks/             # Jupyter notebooks for exploration
├── requirements.txt       # Python dependencies
└── README.md
```

## Model Details

### Features
| Feature | Description | Range |
|---------|-------------|-------|
| years_experience | Professional experience | 0-30 years |
| education_level | 1=High School, 2=Bachelor's, 3=Master's, 4=PhD | 1-4 |
| age | Employee age | 22-60 |
| job_role | 1=Junior, 2=Mid, 3=Senior, 4=Lead, 5=Manager | 1-5 |

### Performance Metrics
- **R² Score**: ~0.95 (explains 95% of salary variance)
- **RMSE**: ~$6,500 (average prediction error)
- **MAE**: ~$5,200 (mean absolute error)

## Technologies Used

- **Python 3.8+**
- **scikit-learn** - Machine learning
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Visualization
- **Streamlit** - Web interface
- **Joblib** - Model persistence

## Key Learnings

- Data preprocessing and feature engineering
- Train/test split methodology
- Model evaluation metrics interpretation
- Building interactive ML applications
- Data visualization best practices

## Future Improvements

- [ ] Add more features (industry, location, skills)
- [ ] Try other algorithms (Random Forest, XGBoost)
- [ ] Deploy to Streamlit Cloud
- [ ] Add API endpoint with FastAPI

## License

MIT License - feel free to use this project for learning purposes.

## Author

Your Name - [GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)
