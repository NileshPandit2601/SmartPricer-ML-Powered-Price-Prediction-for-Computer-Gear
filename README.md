# 💻 SmartPricer: ML-Powered Price Prediction for Computer Gear

SmartPricer is a machine learning project aimed at accurately predicting the selling price of computer components based on product metadata like brand, category, supplier, stock level, and product age. This tool can be used by e-commerce platforms or computer hardware retailers to set optimal prices for their products.

---

## 📊 Problem Statement

Predict the price of computer equipment using product features such as:
- Brand
- Category
- Supplier
- Stock availability
- Date of manufacture (used to calculate product age)

The goal is to automate and optimize pricing decisions using regression models.

---

## 🧹 Data Preprocessing & Feature Engineering

1. **Missing Values**: Dropped rows with missing `Price (USD)` and `date_manufactured`.
2. **Date Parsing**: Extracted `Product Age (Years)` from the manufacturing date.
3. **Target Transformation**: Applied log transformation (`Log_Price = log1p(Price)`) to reduce skewness.
4. **Feature Engineering**:
   - Added polynomial features to capture non-linear effects.
   - Applied one-hot encoding to categorical variables.
   - Standardized numeric features using `StandardScaler`.

---

## 🧠 Model Details

- **Final Model**: Linear Regression
- **Pipeline Components**:
  - PolynomialFeatures (degree = 2)
  - StandardScaler for numeric features
  - OneHotEncoder for categorical features
- **Target Variable**: `Log_Price`

---

## 🔍 Model Selection Process

Several regression models were tested using 5-fold cross-validation:

| Model               | Mean R² Score |
|---------------------|----------------|
| **Linear Regression**   | **0.6873**     |
| Random Forest       | 0.6594         |
| Gradient Boosting   | 0.6561         |
| SVR                 | 0.5834         |
| ElasticNet          | -0.0055        |

**Linear Regression** was selected for its:
- Strong performance
- Simplicity and interpretability
- Stability across different validation sets

---

## 🧪 Final Model Evaluation

Performance on the test set (after full preprocessing):

- **R² Score**: 0.7249
- **MAE**: 0.4750
- **RMSE**: 0.6028

These results indicate a good balance between bias and variance, and reliable predictive power.

---

## 🧠 SQL Queries and Integration

We used **SQLite** in Python (`sqlite3`) to perform exploratory and summary analysis using SQL before applying machine learning. This also shows how data can be used from relational databases in real-life deployment scenarios.

### Sample Queries:

- **1. Total Products by Category**
```sql
SELECT Category, COUNT(*) AS Total_Products
FROM products
GROUP BY Category;
```

- **2. Average Price by Brand**
```sql
SELECT Brand, AVG("Price (USD)") AS Average_Price
FROM products
GROUP BY Brand
ORDER BY Average_Price DESC;
```

- **3. Products with Stock Below Average**
```sql
SELECT *
FROM products
WHERE Stock < (SELECT AVG(Stock) FROM products);
```

- **4. Latest Manufactured Products**
```sql
SELECT * FROM products
ORDER BY "Date Manufactured" DESC
LIMIT 5;
```

These queries were run using pandas’ integration with SQLite for in-notebook analysis.

---

## 📈 Visualizations

- Actual vs Predicted Log Prices  
- Residual Plot  
These plots were created using `matplotlib` and `seaborn` to analyze model behavior and check for overfitting.

---

## 💾 Saved Artifacts

The following files are saved using `joblib` for deployment:
- `price_predictor_pipeline.joblib`: Complete preprocessing and Linear Regression pipeline

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smartpricer.git
   cd smartpricer
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the app (if using Streamlit):
   ```bash
   streamlit run app.py
   ```

4. Or use the trained model in your Python scripts:
   ```python
   import joblib
   model = joblib.load('price_predictor_pipeline.joblib')
   predictions = model.predict(new_data)
   ```

---

## 📦 Requirements

Install them with:
```bash
pip install -r requirements.txt
```

---
## 🚀 Live Demo

Experience **SmartPricer** in action right now:
👉 ## ​ Live Demo
[**smartpricer-ml-powered-price-prediction-for-computer-gear**](https://smartpricer-ml-powered-price-prediction-for-computer-gear-yex8.streamlit.app/)


---
