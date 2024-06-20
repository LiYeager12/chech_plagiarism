import numpy as np
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import re
import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
import streamlit as st

# Download stopwords dan tokenizer dari NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Inisialisasi stemmer dan stopwords
df = pd.read_csv("coba.csv")
df
