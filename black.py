import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Function to extract article text from URL
def extract_article_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract article text
        article_text = ""
        for paragraph in soup.find_all('p'):
            article_text += paragraph.get_text() + "\n"
        return article_text.strip()
    except Exception as e:
        print(f"Error occurred while extracting text from {url}: {e}")
        return ""

# Function to perform sentiment analysis
def sentiment_analysis(text):
    # Load positive and negative words
    positive_words = set(open('positive-words.txt').read().splitlines())
    negative_words = set(open('negative-words.txt').read().splitlines())

    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Calculate positive and negative scores
    positive_score = sum(1 for word in tokens if word in positive_words)
    negative_score = sum(1 for word in tokens if word in negative_words)

    # Calculate polarity and subjectivity scores
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)

    return positive_score, negative_score, polarity_score, subjectivity_score

# Function to compute additional variables
def compute_variables(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Compute word count
    word_count = len(tokens)

    # Compute average sentence length
    sentences = sent_tokenize(text)
    avg_sentence_length = sum(len(word_tokenize(sentence)) for sentence in sentences) / len(sentences)

    # Compute percentage of complex words
    complex_words = [word for word in tokens if len(word) > 7]  # Assuming complex words are longer than 7 characters
    percentage_complex_words = (len(complex_words) / word_count) * 100

    # Compute fog index
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)

    # Compute average number of words per sentence
    avg_words_per_sentence = word_count / len(sentences)

    # Compute complex word count
    complex_word_count = len(complex_words)

    # Compute syllable per word
    syllable_count = sum(len(re.findall(r'[aeiouy]+', word)) for word in tokens)
    syllable_per_word = syllable_count / word_count

    # Compute personal pronouns count
    personal_pronouns = sum(1 for word in tokens if word in {'i', 'we', 'my', 'ours', 'us'})

    # Compute average word length
    avg_word_length = sum(len(word) for word in tokens) / word_count

    return (word_count, avg_sentence_length, percentage_complex_words, fog_index,
            avg_words_per_sentence, complex_word_count, syllable_per_word, personal_pronouns, avg_word_length)

# Define the Streamlit app
def main():
    # Set up Streamlit app title
    st.title("Text Analysis App")

    # Read input Excel file
    input_df = pd.read_excel('c:\\Users\\lokes\\Downloads\\Input.xlsx')

    # Create lists to store computed variables
    positive_scores = []
    negative_scores = []
    polarity_scores = []
    subjectivity_scores = []
    avg_sentence_lengths = []
    percentage_complex_words_list = []
    fog_indexes = []
    avg_words_per_sentence_list = []
    complex_word_counts = []
    word_counts = []
    syllable_per_words = []
    personal_pronouns_list = []
    avg_word_lengths = []

    # Iterate over each row in the input data
    for index, row in input_df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']
        # Extract article text from URL
        article_text = extract_article_text(url)
        # Perform sentiment analysis
        positive_score, negative_score, polarity_score, subjectivity_score = sentiment_analysis(article_text)
        # Compute additional variables
        (word_count, avg_sentence_length, percentage_complex_words, fog_index,
        avg_words_per_sentence, complex_word_count, syllable_per_word, personal_pronouns, avg_word_length) = compute_variables(article_text)
        # Store computed variables
        positive_scores.append(positive_score)
        negative_scores.append(negative_score)
        polarity_scores.append(polarity_score)
        subjectivity_scores.append(subjectivity_score)
        avg_sentence_lengths.append(avg_sentence_length)
        percentage_complex_words_list.append(percentage_complex_words)
        fog_indexes.append(fog_index)
        avg_words_per_sentence_list.append(avg_words_per_sentence)
        complex_word_counts.append(complex_word_count)
        word_counts.append(word_count)
        syllable_per_words.append(syllable_per_word)
        personal_pronouns_list.append(personal_pronouns)
        avg_word_lengths.append(avg_word_length)

    # Create DataFrame to store computed variables
    output_df = pd.DataFrame({
        'URL_ID': input_df['URL_ID'],
        'URL': input_df['URL'],
        'POSITIVE SCORE': positive_scores,
        'NEGATIVE SCORE': negative_scores,
        'POLARITY SCORE': polarity_scores,
        'SUBJECTIVITY SCORE': subjectivity_scores,
        'AVG SENTENCE LENGTH': avg_sentence_lengths,
        'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words_list,
        'FOG INDEX': fog_indexes,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence_list,
        'COMPLEX WORD COUNT': complex_word_counts,
        'WORD COUNT': word_counts,
        'SYLLABLE PER WORD': syllable_per_words,
        'PERSONAL PRONOUNS': personal_pronouns_list,
        'AVG WORD LENGTH': avg_word_lengths
    })

    # Display the output DataFrame in Streamlit
    st.write(output_df)

# Run the Streamlit app
if __name__ == '__main__':
    main()
