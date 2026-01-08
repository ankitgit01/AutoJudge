# 🎯 Coding Problem Difficulty Prediction System

## 📌 Project Overview

Competitive programming platforms host thousands of problems whose difficulty levels are often assigned manually. This process is subjective, time-consuming, and inconsistent across platforms.  

This project presents an **end-to-end machine learning system** that automatically:
- **Classifies coding problems** into *Easy, Medium, or Hard*
- **Predicts a numerical difficulty score** for finer-grained assessment

The system uses **classical machine learning**, **feature engineering**, and **ensemble learning**, and is deployed via an interactive **Streamlit web application**.

---

## 📊 Dataset Used
The dataset provided in the project description is used. Below is the link copy-pasted:
https://github.com/AREEG94FAHAD/TaskComplexityEval-24.git

The dataset consists of programming problems stored in **JSON Lines (`.jsonl`) format**. Each entry includes:

- `title` – Problem title  
- `description` – Full problem statement  
- `input_description` – Input format  
- `output_description` – Output format  
- `sample_io` – Sample input/output  
- `problem_class` – Difficulty label (Easy / Medium / Hard)  
- `problem_score` – Numerical difficulty score  

### Key Characteristics
- Text-heavy dataset
- Class imbalance (Easy problems are more frequent)
- No missing values, but several empty text fields handled during preprocessing

---

## 🧠 Approach & Models Used

### 1️⃣ Preprocessing & Feature Engineering

- Text cleaning and normalization (LaTeX symbols, numbers, whitespace)
- Constraint extraction from input descriptions (log-scaled)
- Text length as a structural feature
- Keyword-based difficulty indicators (Easy / Medium / Hard signals)
- TF-IDF vectorization (unigrams + bigrams)
- Feature scaling and sparse feature stacking

---

### 2️⃣ Classification Models

Multiple classifiers were evaluated:
- Logistic Regression (baseline)
- Random Forest Classifier
- Linear SVM (calibrated)
- Soft Voting Ensemble (LR + RF + SVM)

✅ **Final Classifier:** Random Forest (best balance of accuracy and robustness)

---

### 3️⃣ Regression Models

To predict a continuous difficulty score:

- Random Forest Regressor
- Gradient Boosting Regressor
- Voting Regressor (RF + GB)

#### 🚀 Final Regression Model (Stacked)

- Uses **classifier probability outputs as meta-features**
- Ensemble of **XGBoost Regressor + Ridge Regression**
- Significantly improves MAE and RMSE over baseline models

### Note: Separate models are used for classification and regression tasks; therefore, predicted difficulty classes and problem scores may occasionally appear inconsistent.
---

## 📈 Evaluation Metrics

### Classification
- **Accuracy**
- Confusion Matrix
- Precision, Recall, F1-score

### Regression
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- R² Score

Both models were evaluated on a held-out test set.

---

## 🖥️ Web Interface

The project includes an interactive **Streamlit web application** that allows users to:

- Input a new coding problem (title, description, input/output)
- View predicted difficulty class (color-coded)
- View predicted numerical difficulty score (out of 10)
- Inspect class probability distribution

The web app strictly reuses the same preprocessing pipeline and trained models to ensure consistency with offline results.

---

## ▶️ Steps to Run the Project Locally

### 1️⃣ Clone the Repository
```bash
git clone <your-github-repo-url>
cd <repo-folder>
```

### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib xgboost
```

### 4️⃣ Run the Streamlit App
*Ensure that the app.py is present directly in the current working directory or use the complete path of app.py*
```bash
python -m streamlit run app.py
```

The application will open in your browser at:
```
http://localhost:8501
```

---

## 🎥 Demo Video

📽️ **project demo video Youtube link:**  
https://youtu.be/W6ff9ikGGGo

📽️ **Alternative link (Google drive) if the above one doesn't work** 


---

## 📂 Project Structure

```
├── app.py
├── pickle/
│   ├── final_classifier_58.pkl
│   ├── final_regressor.pkl
│   ├── tfidf_vectorizer.pkl
│   └── numeric_scaler.pkl
├── preprocessed/
│   ├── X_final.npz
│   ├── y.csv
│   └── y_score.csv
├── problems_data.jsonl
├── README.md
```

---

## 👤 Author

**Ankit**  
B.Tech Student  
Interests: Data Science, Machine Learning, Artificial Intelligence

