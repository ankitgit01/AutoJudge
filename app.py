import streamlit as st
import joblib
import numpy as np
import re
import math
from scipy.sparse import hstack, csr_matrix
import os

# python -m streamlit run app.py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Coding Problem Difficulty Predictor",
    page_icon="🎯",
    layout="centered"
)

# Load trained models and transformers
@st.cache_resource
def load_models():
    clf = joblib.load(os.path.join(BASE_DIR, "pickle/final_classifier_58.pkl"))
    reg = joblib.load(os.path.join(BASE_DIR, "pickle/final_regressor.pkl"))
    tfidf = joblib.load(os.path.join(BASE_DIR, "pickle/tfidf_vectorizer.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "pickle/numeric_scaler.pkl"))
    return clf, reg, tfidf, scaler


clf, reg, tfidf, scaler = load_models()

# Text cleaning
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.replace('$', ' ')

    replacements = {
        r'\\le': '<=', r'\\ge': '>=', r'\\lt': '<', r'\\gt': '>',
        r'\\neq': '!=', r'\\times': '*', r'\\dots': '...',
        r'\\': ' ', r'\n': ' '
    }

    for pat, repl in replacements.items():
        text = re.sub(pat, repl, text)

    while re.search(r'(\d)\s?,\s?(\d)', text):
        text = re.sub(r'(\d)\s?,\s?(\d)', r'\1\2', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_log_constraint(text):
    if not isinstance(text, str) or not text:
        return 0

    values = []

    powers = re.findall(r'(\d+)\s*\^\s*\{?(\d+)\}?', text)
    for base, exp in powers:
        try:
            val = float(base) ** float(exp)
            if val < 1e20:
                values.append(val)
        except:
            pass

    integers = re.findall(r'\b\d+\b', text)
    for x in integers:
        try:
            values.append(float(x))
        except:
            pass

    relevant = [v for v in values if 10 < v < 1e19]
    if not relevant:
        return 0

    return math.log(max(relevant))


# Keyword features
KEYWORD_FEATURES = {
    'easy_signals': ['swap', 'reverse', 'palindrome', 'even', 'odd', 'sort', 'min', 'max'],
    'medium_signals': ['dynamic programming', 'dp', 'dijkstra', 'bfs', 'dfs',
                       'greedy', 'binary search', 'modulo', 'prime', 'xor'],
    'hard_signals': ['segment tree', 'bitmask', 'flow', 'matching',
                     'centroid', 'heavy light', 'convex hull', 'fft']
}

def extract_keyword_features(text):
    text = text.lower()
    features = []
    for category in ['easy_signals', 'medium_signals', 'hard_signals']:
        for word in KEYWORD_FEATURES[category]:
            features.append(1 if word in text else 0)
    return features



def preprocess_input(title, desc, input_desc, output_desc):

    input_desc = input_desc if input_desc.strip() else desc
    output_desc = output_desc if output_desc.strip() else desc

    clean_input = clean_text(input_desc)
    clean_desc = clean_text(desc)

    combined_text = f"{title} {clean_input} {clean_desc}"

    # TF-IDF
    X_text = tfidf.transform([combined_text])

    # Numeric features
    log_constraint = extract_log_constraint(clean_input)
    text_len = len(combined_text)
    X_numeric = scaler.transform([[log_constraint, text_len]])

    # Keyword features
    keyword_vec = np.array([extract_keyword_features(combined_text)])
    keyword_sparse = csr_matrix(keyword_vec)

    # Final feature vector
    X_final = hstack([X_text, X_numeric, keyword_sparse])

    return X_final



st.title("🎯 Coding Problem Difficulty Predictor")

st.markdown(
    "Predict the **difficulty class** and **difficulty score** of a coding problem "
    "using classical machine learning and feature engineering."
)

with st.form("prediction_form"):
    title = st.text_input("Title", placeholder="e.g. Longest Special Path")
    desc = st.text_area("Problem Description", height=160)
    input_desc = st.text_area("Input Description", height=100)
    output_desc = st.text_area("Output Description", height=100)

    submitted = st.form_submit_button("🔮 Predict Difficulty")


if submitted:
    if not title.strip() or not desc.strip():
        st.error("Please provide at least a title and problem description.")
    else:
        with st.spinner("Analyzing problem..."):
            X = preprocess_input(title, desc, input_desc, output_desc)

            pred_class = clf.predict(X)[0]
            class_probs = clf.predict_proba(X)

            X_reg = hstack([X, class_probs])
            score = float(reg.predict(X_reg)[0])

        color = "🟢" if pred_class == "easy" else "🟡" if pred_class == "medium" else "🔴"

        st.success("Prediction complete!")

        col1, col2 = st.columns(2)
        col1.metric("Difficulty Class", f"{color} {pred_class.capitalize()}")
        col2.metric("Difficulty Score", f"{round(score, 2)} / 10")

        with st.expander("Class Probabilities"):
            for label, prob in zip(clf.classes_, class_probs[0]):
                st.write(f"**{label.capitalize()}**: {prob:.3f}")


st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#777;'>"
    "Streamlit • TF-IDF • Feature Engineering • Random Forest • XGBoost • Ridge"
    "</div>",
    unsafe_allow_html=True
)
