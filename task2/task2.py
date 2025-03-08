import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Download VADER for sentiment analysis
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z ]+', '', text)  # Remove special characters
    return text.lower()

# Function to analyze sentiment
def analyze_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to read CSV file and process tweets
def process_tweets(file_path):
    df = pd.read_csv(file_path)
    df = df[['Timestamp', 'Text']]  # Ensure correct column names
    df['Cleaned_Text'] = df['Text'].apply(clean_text)
    df['Sentiment'] = df['Cleaned_Text'].apply(analyze_sentiment)
    return df

# Main function
def main():
    file_path = "twitter_dataset.csv"  # Change this to your dataset file
    tweets_df = process_tweets(file_path)

    # Print sample data
    print(tweets_df.head())

    # Plot sentiment distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Sentiment', data=tweets_df, palette='coolwarm')
    plt.title('Sentiment Analysis from CSV Data')
    plt.show()

if __name__ == '__main__':
    main()
