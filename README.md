# Medication Review Condition Predictor

A simple Streamlit web app that predicts the health **condition** associated with a medication review, based on the review text. The model was trained on medication review data using scikit-learn.

## Features

- Paste or type a medication review and get an instant prediction
- Clear display of the predicted condition
- Confidence scores for all supported conditions

## Supported conditions

The classifier predicts one of:

- Acne
- Anxiety
- Birth Control
- Depression
- Pain

## Model

| Component   | Type                                      |
| ----------- | ----------------------------------------- |
| Vectorizer  | `sklearn.feature_extraction.text.CountVectorizer` |
| Classifier  | `sklearn.naive_bayes.MultinomialNB`       |

Artifacts are loaded from `vectorizer.pkl` and `model.pkl`. Review text is transformed with the vectorizer (`transform`, not `fit_transform`) before prediction.

## Project structure

```
Medication-Review-Condition-Predictor/
├── app.py              # Streamlit application
├── model.pkl           # Trained classifier
├── vectorizer.pkl      # Fitted text vectorizer
├── requirements.txt    # Python dependencies
└── README.md
```

## Requirements

- Python 3.9+

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/reventhTheCoder333/Medication-Review-Condition-Predictor.git
   cd Medication-Review-Condition-Predictor
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the app

```bash
streamlit run app.py
```

If `streamlit` is not on your PATH:

```bash
python -m streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`), enter a review, and click **Predict**.

## Example

Try a review like:

> This pill helped my acne clear up after a few weeks with minimal side effects.

The app should predict **Acne** with high confidence.

## License

This project is provided as-is for educational and demonstration purposes.
