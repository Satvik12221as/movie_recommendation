import pandas as pd

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = movies.merge(credits, on="title")
movies = movies[["movie_id", "title", "overview", "cast", "crew"]]
movies.dropna(inplace=True)                                                # using dropna to remove missing values(NAN)

movies.dropna(inplace=True)                                                # using dropna to remove missing values(NAN)

tfidf = TfidfVectorizer(stop_words="english")                              #Changing the overview into vectors
vectors = tfidf.fit_transform(movies["overview"])

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

similarity = cosine_similarity(vectors)

def recommend(movie_name):
    index = movies[movies["title"] == movie_name].index[0]                 #Find index of the movie
    distances = list(enumerate(similarity[index]))                         #list of similar scores
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6] #Sorting movies
    for i in movies_list:                                                  #Printing top 5 because of the range [1:6]
        print(movies.iloc[i[0]].title)
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6] #Sorting movies
    for i in movies_list:                                                  #Printing top 5 because of the range [1:6]
        print(movies.iloc[i[0]].title)
recommend("The Lego Movie")
print("These are the similar movies to The Lego Movie")

# Refine the data preprocessing to handle missing values more effectively
movies['overview'] = movies['overview'].fillna('')

# Experiment with different feature extraction techniques such as word embeddings
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(movies["overview"])
similarity = cosine_similarity(vectors)

# Tune the hyperparameters of the cosine similarity function
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors, dense_output=False)
recommend("Spectre")
print("These are the similar movies to Spectre")
recommend("The Lego Movie")
print("These are the similar movies to The Lego Movie")