import pandas as pd

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = movies.merge(credits, on="title")


movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast"]]


from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

def combine_features(row):
    return row["overview"] + " " + row["genres"] + " " + row["keywords"] + " " + row["cast"]

movies["combined_features"] = movies.apply(combine_features, axis=1)

sentences = [row.split() for row in movies["combined_features"]]
model = Word2Vec(sentences, vector_size=100, min_count=1)

def get_vector(row):
    words = row.split()
    vector = np.zeros(100)
    for word in words:
        if word in model.wv:
            vector += model.wv[word]
    return vector / len(words)

movies["vector"] = movies["combined_features"].apply(get_vector)


similarity = cosine_similarity(movies["vector"].tolist())
similarity = cosine_similarity(vectors)

def recommend(movie_name):
    
    index = movies[movies["title"] == movie_name].index[0]                 #Find index of the movie
    distances = list(enumerate(similarity[index]))                         #list of similar scores
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6] #Sorting movies
    for i in movies_list:                                                  #Printing top 5 because of the range [1:6]
        print(movies.iloc[i[0]].title)

# Example usage
recommend("Spectre")
print("These are the similar movies to Spectre")
recommend("The Lego Movie")
print("These are the similar movies to The Lego Movie")
print("These are the similar movies to Spectre")
print("These are the similar movies to The Lego Movie")