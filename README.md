# 🏠 Mumbai House Price Prediction Using Machine Learning

## 📌 Project Overview

Mumbai House Price Prediction is a Machine Learning regression project that predicts the estimated prices of residential properties in Mumbai based on property characteristics such as BHK, property type, locality, area, region, construction status, and property age.

The project uses the K-Nearest Neighbors (KNN) Regression algorithm to learn patterns from historical property data and estimate house prices.

A Streamlit web application is developed to allow users to enter property details and obtain estimated house prices.

## 🎯 Problem Statement

House prices in Mumbai vary significantly depending on location, property size, number of bedrooms, property type, construction status, and property age.

The objective of this project is to develop a Machine Learning regression model that predicts residential property prices using historical housing data.

## 🎯 Target Variable

- **Target Variable:** `price_lakhs`
- **Problem Type:** Regression
- **Target Description:** Estimated residential property price expressed in Indian lakhs.

The original dataset contains prices in two units: Crores (Cr) and Lakhs (L). These values were converted into a common unit (Lakhs) to create a consistent numerical target variable.

## 📂 Dataset Information

- **Dataset:** Mumbai House Prices
- **Total Records After Duplicate Removal:** 55,726
- **Original Features:** 9
- **Target Variable:** `price_lakhs`
- **Data Format:** CSV

### Dataset Features

| Feature | Description |
|---|---|
| `bhk` | Number of bedrooms |
| `type` | Property type |
| `locality` | Property locality |
| `area` | Property area |
| `price` | Original property price |
| `price_unit` | Price unit (Cr or L) |
| `region` | Property region |
| `status` | Construction status |
| `age` | Property age |

### Target Transformation

Original prices were converted into lakhs:

- 2.50 Cr → 250 Lakhs
- 52.51 L → 52.51 Lakhs
- 1.73 Cr → 173 Lakhs

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Dataset loading and initial inspection.
2. Missing value analysis.
3. Duplicate record identification and removal.
4. Categorical feature analysis.
5. Price conversion from Crores and Lakhs into a common unit.
6. Outlier analysis.
7. Numerical and categorical feature identification.
8. Feature selection.
9. Training and testing data separation.
10. Numerical feature scaling.
11. Categorical feature encoding using One-Hot Encoding.
12. Locality frequency encoding.

Potential outliers were analyzed and retained because some high property prices and large property areas may represent genuine luxury properties.

## 🔍 Features Used for Prediction

The final model uses seven input features:

- `bhk`
- `type`
- `locality`
- `area`
- `region`
- `status`
- `age`

The original `price` and `price_unit` columns are excluded from the model inputs to prevent target leakage.

## 🤖 Machine Learning Algorithms

The project explored multiple individual regression algorithms:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Decision Tree Regression
- K-Nearest Neighbors (KNN) Regression
- Support Vector Regression (SVR)

KNN and Decision Tree were further evaluated using a corrected preprocessing pipeline and cross-validation.

**Final Selected Algorithm:** K-Nearest Neighbors (KNN) Regression.

**Optimal K Value:** 3

The optimal K value was selected using 5-fold cross-validation by comparing different neighbor values.

No ensemble algorithms were used.

## 📊 Model Training

The dataset was divided into training and testing sets:

| Dataset | Records |
|---|---:|
| Training Data | 44,580 |
| Testing Data | 11,146 |
| Total | 55,726 |

- Training percentage: 80%
- Testing percentage: 20%
- Random state: 42

A preprocessing pipeline was used to apply numerical scaling, categorical encoding, and locality frequency encoding.

## 📈 Model Evaluation Metrics

The final KNN model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Final KNN Model Results

| Metric | Result |
|---|---:|
| MAE | 31.73 |
| MSE | 9593.19 |
| RMSE | 97.94 |
| R² Score | 0.8109 |

**R² Score: 81.09%**

The final model achieved an R² score of 0.8109 on the unseen test dataset.

## 🔎 Additional Model Analysis

The following analyses were also performed:

- Actual vs Predicted Price Visualization
- Residual Analysis
- Price Category-wise Error Analysis
- Underprediction and Overprediction Analysis
- Percentage Prediction Error Analysis
- Largest Prediction Error Analysis
- Permutation Feature Importance

These analyses help understand model performance and identify areas where prediction accuracy can be improved.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## 📁 Project Structure

```text
House_Price_Prediction/
│
├── Mumbai_House_Prices.csv
│
├── notebooks/
│   └── Mumbai_House_Price_Prediction.ipynb
│
├── models/
│   └── mumbai_house_price_knn.pkl
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd House_Price_Prediction
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 🌐 Streamlit Application

The Streamlit application allows users to:

1. Select the number of bedrooms (BHK).
2. Choose the property type.
3. Select the locality.
4. Enter the property area.
5. Select the region.
6. Choose the construction status.
7. Select the property age.
8. Generate the predicted house price.

The predicted price is displayed in both Indian Lakhs and Crores.

## 🚀 Deployment

The application can be deployed using Streamlit Community Cloud.

Deployment steps:

1. Upload the project files to a GitHub repository.
2. Ensure that `app.py`, `requirements.txt`, the dataset, and the trained model are included.
3. Open Streamlit Community Cloud.
4. Connect your GitHub repository.
5. Select `app.py` as the main application file.
6. Deploy the application.

## ⚠️ Limitations

- Predictions depend on the quality and coverage of the historical dataset.
- Luxury properties have higher prediction errors than budget and mid-range properties.
- Property prices may change due to market conditions and other external factors.
- The model provides estimated prices and should not be considered an official property valuation.

## 🔮 Future Enhancements

- Improve prediction accuracy for luxury properties.
- Explore additional location-related features.
- Evaluate further regression algorithms.
- Improve the Streamlit interface and user experience.
- Incorporate more recent housing market data.
- Deploy the application for public access.

## 👨‍💻 Conclusion

This project demonstrates an end-to-end Machine Learning regression workflow for predicting residential property prices in Mumbai.

It includes data cleaning, exploratory data analysis, feature engineering, preprocessing, model training, cross-validation, model evaluation, error analysis, and Streamlit deployment.

The final KNN Regression model achieved an R² score of 81.09% on the test dataset and was integrated into a Streamlit application for interactive house price prediction.
