# Nigeria-Housing-Price-Prediction
Predicting Housing Prices in Nigeria Analyzing Traditional Functional Forms and Machine Learning Techniques . This study investigates the effectiveness of various functional forms and machine learning techniques for predicting housing prices in Nigeria, using property features and location data.

### Predicting Housing Prices in Nigeria with Traditional Functional Forms and Machine Learning

---

## Project Description

This project aims to develop a predictive model for housing prices in Nigeria using a combination of traditional regression models and machine learning techniques. By leveraging property features and location data, we investigate the accuracy and practicality of different methods, including functional transformations (e.g., linear, quadratic) and advanced machine learning models, to determine which approach best predicts housing prices in Nigeria.

The project will culminate in a user-friendly Streamlit app that allows users to input housing features to obtain price predictions based on the selected model.

---

## Project Objectives

1. **Predict Housing Prices**: Use various traditional and machine learning models to accurately predict housing prices based on Nigerian market data.
2. **Analyze Effectiveness of Models**: Compare traditional functional forms (e.g., linear, quadratic) against machine learning models like Random Forest and Gradient Boosting to assess the most effective approach.
3. **Develop a Deployment-Ready App**: Build a Streamlit application to allow users to interact with the model and predict housing prices based on inputted features.

---

## Data Overview

- **Source**: Housing data includes information on property features and location details, capturing attributes like property type, number of rooms, size, age, and location.
- **Data Processing**: The data will be cleaned, preprocessed, and transformed for effective model training and validation.
- **Features**:
  - **Property Characteristics**: Number of bedrooms, bathrooms, square footage, property type, age of property.
  - **Location Features**: Regional or neighborhood indicators to capture geographical price trends.

---

## Methodology

1. **Data Preprocessing and Feature Engineering**: Handle missing values, outliers, and encode categorical variables. Use transformations (e.g., Box-Cox) for skewed features to improve model performance.
2. **Modeling**:
   - **Traditional Regression Models**: Test functional forms, such as linear, quadratic, and logarithmic regressions.
   - **Machine Learning Models**: Train models like Random Forest and Gradient Boosting Regressor to evaluate performance.
3. **Model Evaluation**: Compare models using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
4. **Deployment**: Deploy the best-performing model using Streamlit, creating an interactive interface for users to input property data and receive price predictions.

---

## Setup Instructions

Follow these steps to set up the project locally.

### Prerequisites

- Python 3.8+
- Git
- Virtual Environment (optional but recommended)

### Installation

1. **Clone the repository**:
   ```bash```
   git clone https://github.com/Johnnysnipes90/Nigeria-Housing-Price-Prediction.git
   cd Nigeria-Housing-Price-Prediction
2. **Create a virtual environment**
   python3 -m venv venv
  source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. **Install the required packages**
   pip install -r requirements.txt
4. **Download the data**
   - Download the raw data from the data/ directory

## Running the Project
**Exploratory Data Analysis**

Open the Jupyter notebooks in the notebooks/ folder to explore the data and initial findings.
**Model Training**

Run the scripts in the src/ folder to train and evaluate models.
**Launch Streamlit App**
streamlit run app.py

## Expected Outcomes
Model Evaluation: Insights into the effectiveness of different model types for Nigerian housing price prediction.
Final Application: A Streamlit app allowing users to predict housing prices by entering property characteristics.

## Contributing
Feel free to fork this repository and make pull requests with suggestions or improvements. Please ensure that you test any code changes locally before submitting.


