from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models
sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
classification = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the categories for classification
categories = ["Compensation and Benefits", "Work Life Balance", "Job Security", "Culture", "Career Path"]

@app.route("/", methods=["GET"])
def home():
    return "Sentiment Analysis and Classification Service is running."

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Parse JSON input
        data = request.get_json()
        sentences = data.get("sentences", [])

        if not sentences:
            return jsonify({"error": "No sentences provided"}), 400

        # Initialize results list
        results = []

        # Process each sentence
        for sentence in sentences:
            # Perform sentiment analysis
            sentiment_result = sentiment_analysis(sentence)
            sentiment = sentiment_result[0]["label"]
            sentiment_confidence = sentiment_result[0]["score"] * 100

            # Perform zero-shot classification
            classification_result = classification(sentence, candidate_labels=categories)
            classifications = {label: score * 100 for label, score in zip(classification_result["labels"], classification_result["scores"])}

            # Store results
            result = {
                "sentence": sentence.strip(),
                "sentiment": sentiment,
                "confidence": sentiment_confidence,
                "classifications": classifications
            }
            results.append(result)

        # Return results as JSON
        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
