import os
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
from concurrent.futures import ThreadPoolExecutor


# Set NLTK data path dynamically based on the current working directory
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

# Set NLTK data path
nltk.data.path.append('/Users/amadoudiakhadiop/Documents/chatbot/nltk_data')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words


# Function to preprocess HTML content
def preprocess_html(html_content):
    soup = BeautifulSoup(html_content, 'lxml')
    # Extract text content from HTML
    text_content = soup.get_text(separator=' ')
    # Tokenize the text into sentences
    sentences = sent_tokenize(text_content)
    # Preprocess each sentence
    corpus = [preprocess(sentence) for sentence in sentences]
    return corpus


# Load HTML files from a folder
def load_html_files(folder_path):
    html_files = [f for f in os.listdir(folder_path) if f.endswith('.html')]
    corpus = []
    for file in html_files:
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
            html_content = f.read()
            corpus.extend(preprocess_html(html_content))
    return corpus


# Load the text from HTML files and preprocess the data
corpus = load_html_files('untitled folder')
# Transform the corpus into a set of unique words for each sentence
unique_word_sets = [set(sentence) for sentence in corpus]
print("DONE")

# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)

    print("Processed query:", query)  # Add this line for debugging

    # Compute the similarity between the query and each sentence in the text
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence, unique_words in zip(corpus, unique_word_sets):
        similarity = len(set(query).intersection(unique_words)) / float(len(set(query).union(unique_words)))
        print(f"Similarity with sentence {sentence}: {similarity}")  # Add this line for debugging
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)

    print("Most relevant sentence:", most_relevant_sentence)  # Add this line for debugging
    return most_relevant_sentence

print("DONE")


def chatbot(question):
    # Find the most relevant sentence
    most_relevant_sentence = get_most_relevant_sentence(question)
    # Return the answer
    return most_relevant_sentence
print("DONE")


# Create a Streamlit app
def main():
    st.title("Chatbot")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")

    # Get the user's question using st.text_area
    question = st.text_area("You:", height=100)

    # Check if Enter key (carriage return) is pressed
    if question.endswith('\n'):
        # Remove the trailing newline character
        question = question.rstrip('\n')

        # Call the chatbot function with the question and display the response
        response = chatbot(question)
        st.write("Chatbot:", response)
print("DONE")


# Load the text from HTML files and preprocess the data
html_files = ['/Users/amadoudiakhadiop/Desktop/untitled folder']
with ThreadPoolExecutor() as executor:
    results = list(executor.map(preprocess_html, html_files))
print("DONE")


if __name__ == "__main__":
    main()


