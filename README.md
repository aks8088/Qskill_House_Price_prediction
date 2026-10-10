# House Price Prediction Using Machine Learning

## Project Overview

This project predicts California housing values using supervised machine learning regression algorithms. It demonstrates dataset loading, data preprocessing, model training, model evaluation, and visualization.

## Dataset

- **Name:** California Housing Dataset
- **Source:** Scikit-learn
- **Loading method:** `fetch_california_housing(as_frame=True)`
- **Target:** Median house value, converted to a dollar-scale value by multiplying by 100,000.

The dataset is fetched automatically by Scikit-learn when the script runs. An internet connection may be required on the first run.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Machine Learning Models

- Linear Regression
- Ridge Regression
- Random Forest Regressor

The models are compared using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

## Project Structure

```text
house_price_prediction/
├── results/
├── models/
├── house_price_prediction.py
├── compare_models.py
├── finalize_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## How to Run

Run the baseline model:

```bash
python house_price_prediction.py
```

Compare regression models:

```bash
python compare_models.py
```

Train and save the selected model:

```bash
python finalize_model.py
```

## Evaluation

The project reports MAE, RMSE, and R². Lower MAE and RMSE indicate smaller prediction errors, while a higher R² generally indicates a better fit.

The actual results are saved in the `results/` directory after execution.

## Limitations

The dataset contains historical California housing data. Its predictions should not be interpreted as current market valuations for individual properties.

## Author

QSkill AI/ML Internship Project
