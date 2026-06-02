from pathlib import Path

import pickle
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


@st.cache_resource
def load_artifacts():
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return vectorizer, model


def predict_condition(review_text: str) -> tuple[str, dict[str, float] | None]:
    vectorizer, model = load_artifacts()
    features = vectorizer.transform([review_text])
    prediction = model.predict(features)[0]
    probabilities = None
    if hasattr(model, "predict_proba"):
        classes = model.classes_
        scores = model.predict_proba(features)[0]
        probabilities = dict(zip(classes, scores))
    return prediction, probabilities


st.set_page_config(
    page_title="Medication Review Condition Predictor",
    page_icon="💊",
    layout="centered",
)

st.title("Medication Review Condition Predictor")
st.caption(
    "Enter a medication review below. The model will predict which health "
    "condition the review is most likely about."
)

review = st.text_area(
    "Your review",
    placeholder="Example: This medication really helped clear my skin after a few weeks...",
    height=160,
    label_visibility="collapsed",
)

if st.button("Predict", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review before predicting.")
    else:
        with st.spinner("Analyzing review..."):
            condition, probabilities = predict_condition(review.strip())

        st.success("Predicted condition")
        st.markdown(f"## {condition}")

        if probabilities:
            st.subheader("Confidence by condition")
            sorted_probs = sorted(
                probabilities.items(), key=lambda item: item[1], reverse=True
            )
            for label, score in sorted_probs:
                st.progress(float(score), text=f"{label}: {score:.1%}")
