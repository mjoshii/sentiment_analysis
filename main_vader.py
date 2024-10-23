import polars as pl
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast

analyser = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    return str(analyser.polarity_scores(text))

def getTextAnalysis(n):
    """
    The function `getTextAnalysis` categorizes a number as negative, neutral, or positive based on its
    value.
    
    :param n: The function `getTextAnalysis(n)` takes a numerical input `n` and returns a text analysis
    based on the value of `n`. If `n` is less than 0, it returns "negative". If `n` is equal to 0, it
    returns "neutral". Otherwise, if
    :return: The function `getTextAnalysis` returns a string indicating whether the input number `n` is
    negative, neutral (zero), or positive.
    """
    if n < 0:
        return "negative"
    elif n == 0:
        return "neutral"
    else:
        return "positive"


# Encoding is required to ensure we don't have false "null" values
df = pl.read_csv("sentiment_feedback.csv", ignore_errors=True, encoding='latin-1')
# df = df.drop(columns=['File Name', '_id', 'sentiment', 'sentimentMagnitude', 'url', 'variantId'])

# Concatenating the feedback to perform a consolidated analysis on the sentiment of the reviewer
df = df.with_columns([pl.concat_str([pl.col("title"), pl.col("feedback").str.replace_all("Pros:", " ").str.replace_all("Cons:", " ")], 
                                    separator=" ").alias("consolidated_feedback")
                                    ])

# Remove the original 'feedback' and 'title' columns, not required
df = df.drop(["feedback", "title"])
# print(df.describe())
# print(df['consolidated_feedback'][0])

# Calculate the subjectivity and polarity for each review which is then used to evaluate the sentiment of the reviewer
df = df.with_columns(pl.col('consolidated_feedback').apply(analyze_sentiment).alias('compound_scores'))
df = df.with_columns(pl.col('compound_scores').str.split("'compound':").list.get(1).str.strip('}').str.strip().alias('compound'))
df = df.with_columns(pl.col('compound').cast(pl.Float64).alias('compound'))

df = df.with_columns(df['compound'].apply(getTextAnalysis).alias('score'))
# print(df.head())

positive = df.filter(pl.col('score') == 'positive')
negative = df.filter(pl.col('score') == 'negative')
neutral = df.filter(pl.col('score') == 'neutral')
print(str(positive.shape[0]/(df.shape[0])*100) + " % of positive reviews")
print(str(negative.shape[0]/(df.shape[0])*100) + " % of negative reviews")
print(str(neutral.shape[0]/(df.shape[0])*100) + " % of neutral reviews")

# print(df.select(pl.col('score'), pl.col('rating')))

# # Creating a word cloud
# positive_words = ' '.join([word for word in df.filter(pl.col('score') == 'positive')['consolidated_feedback']])
# positive_wordCloud = WordCloud(width=600, height=400).generate(positive_words)
# plt.title('Positive Sentiments')
# plt.imshow(positive_wordCloud)
# plt.show()

# negative_words = ' '.join([word for word in df.filter(pl.col('score') == 'negative')['consolidated_feedback']])
# negative_wordCloud = WordCloud(width=600, height=400).generate(negative_words)
# plt.title('Negative Sentiments')
# plt.imshow(positive_wordCloud)
# plt.show()

# neutral_words = ' '.join([word for word in df.filter(pl.col('score') == 'neutral')['consolidated_feedback']])
# neutral_wordCloud = WordCloud(width=600, height=400).generate(neutral_words)
# plt.title('Neutral Sentiments')
# plt.imshow(neutral_wordCloud)
# plt.show()

df.write_csv('results_vader.csv')