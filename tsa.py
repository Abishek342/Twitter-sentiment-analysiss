import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from scipy.special import softmax

# Load model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Preprocess tweet
def preprocess_tweet(tweet):
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    return " ".join(tweet_words)

# Perform sentiment analysis
def get_sentiment(tweet):
    model, tokenizer = load_model()
    tweet_proc = preprocess_tweet(tweet)
    
    # Encode the tweet
    encoded_tweet = tokenizer(tweet_proc, return_tensors='tf')
    output = model(**encoded_tweet)
    
    # Get scores and apply softmax
    scores = output[0][0].numpy()
    scores = softmax(scores)

    labels = ['Negative', 'Neutral', 'Positive']
    sentiment_scores = {labels[i]: scores[i] for i in range(len(scores))}
    return sentiment_scores

# Streamlit app
st.title('Twitter Sentiment Analysis')

# Input from user
tweet_input = st.text_area('Enter the Tweet')

if st.button('Analyze Sentiment'):
    if tweet_input:
        sentiment_result = get_sentiment(tweet_input)
        st.write("Sentiment Scores:")
        for sentiment, score in sentiment_result.items():
            st.write(f"{sentiment}: {score:.4f}")
    else:
        st.write("Please enter a tweet.")
