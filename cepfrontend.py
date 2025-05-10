import streamlit as st
import joblib

#Load the models
svm_model = joblib.load('svm_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_news_LR(news_title, news_text):
    content = news_title + " " + news_text
    content_tfidf = vectorizer.transform([content])
    prediction = lr_model.predict(content_tfidf)
    probability = lr_model.predict_proba(content_tfidf)

    return {
        "prediction": "Fake" if prediction[0] == 1 else "Real",
        "probability": round(probability[0][prediction[0]] * 100, 2)
    }
def predict_news_SVM(news_title, news_text):
    content = news_title + " " + news_text
    content_tfidf = vectorizer.transform([content])
    prediction = svm_model.predict(content_tfidf)
    probability = svm_model.predict_proba(content_tfidf)
    return {
        "prediction": "Fake" if prediction[0] == 1 else "Real",
        "probability": round(probability[0][prediction[0]] * 100, 2)
    }



st.set_page_config(page_title="Fake News Classifier", layout="centered")

st.title("📰 Fake News Detection App")
st.subheader("Classify news articles using ML models")

# Input Fields
title = st.text_input("Enter the news title")
text = st.text_area("Enter the news content", height=200, )

# Prediction Result Placeholder
result = None

# Prediction Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Predict using SVM"):
        if title.strip() and text.strip():
            pred = predict_news_SVM(title, text)
            result = f"Prediction using SVM: **{pred['prediction']}** ({pred['probability']}% confidence)"
        else:
            st.warning("Please fill both title and text fields.")

with col2:
    if st.button("Predict using Logistic Regression"):
        if title.strip() and text.strip():
            pred = predict_news_LR(title, text)
            result = f"Prediction using Logistic Regression: **{pred['prediction']}** ({pred['probability']}% confidence)"
        else:
            st.warning("Please fill both title and text fields.")

# Display Result
if result:
    st.markdown("---")
    st.success(result)
