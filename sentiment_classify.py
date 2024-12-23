import pandas as pd
from transformers import pipeline
import re

# Load the pre-trained sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
classification = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the categories for classification
categories = ["Compensation and Benefits", "Work Life Balance", "Job Security", "Culture", "Career Path"]

# Load the CSV file
file_path = "sentiment_feedback_new.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Clean the feedback column by removing "Pros:" and "Cons:"
def clean_feedback(text):
    text = re.sub(r"\bPros:\s*", "", str(text), flags=re.IGNORECASE)
    text = re.sub(r"\bCons:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()

data['feedback_cleaned'] = data['feedback'].apply(clean_feedback)

# Split long feedback into chunks while respecting sentence boundaries
def split_text_into_chunks(text, max_length=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split on sentence boundaries
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if chunk.strip()]  # Remove empty chunks

# Initialize results list
results = []

# Process each feedback
for idx, feedback in enumerate(data['feedback_cleaned']):
    if not feedback.strip():  # Skip blank or empty feedback
        continue

    chunks = split_text_into_chunks(feedback, max_length=500)
    sentiment_aggregated = {"POSITIVE": 0, "NEGATIVE": 0}
    classifications_aggregated = {category: 0 for category in categories}

    for chunk in chunks:
        if not chunk.strip():  # Skip empty chunks
            continue

        # Perform sentiment analysis
        sentiment_result = sentiment_analysis(chunk)
        sentiment = sentiment_result[0]["label"]
        sentiment_confidence = sentiment_result[0]["score"]
        sentiment_aggregated[sentiment] += sentiment_confidence

        # Perform zero-shot classification
        classification_result = classification(chunk, candidate_labels=categories)
        for label, score in zip(classification_result["labels"], classification_result["scores"]):
            classifications_aggregated[label] += score

    # Normalize aggregated scores
    total_chunks = len(chunks)
    sentiment = max(sentiment_aggregated, key=sentiment_aggregated.get)
    sentiment_confidence = sentiment_aggregated[sentiment] / total_chunks * 100
    classifications_normalized = {label: score / total_chunks * 100 for label, score in classifications_aggregated.items()}

    # Append results
    results.append(
        [data.loc[idx, "id"], feedback, sentiment, f"{sentiment_confidence:.2f}%"]
        + [f"{classifications_normalized[category]:.2f}%" for category in categories]
    )

# Prepare headers
headers = ["id", "Feedback", "Sentiment", "Confidence"] + categories

# Save results to a new CSV
output_csv_path = "sentiment_classification_results2.csv"
output_df = pd.DataFrame(results, columns=headers)
output_df.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")
