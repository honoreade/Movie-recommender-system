
# import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import pickle

# movies = pd.read_csv("25k IMDb movie Dataset.csv")
# movies = movies[:10000] 

# movies['index'] = movies.index

# movies_features = movies["movie title"]+' '+movies["Generes"]+' '+movies["Overview"]+' '+movies["Plot Kyeword"]+' '+movies["Director"]+' '+movies["Top 5 Casts"]+' '+movies["Writer"]

# # # converting the text data to feature vectors
# vectorizer = TfidfVectorizer()
# features_vectors = vectorizer.fit_transform(movies_features.values.astype('U'))
# print(features_vectors)

# # # getting the similarity scores using cosine similarity
# similarity = cosine_similarity(features_vectors)
# print(similarity)


# Load pretrained Model
with open('Movie_recommend.pkl', 'rb') as file:
    vectorizer,similarity,movies = pickle.load(file)

# movie_name = input(' Enter your favourite movie name : ')
movie_name = "avatar"

list_of_all_titles = movies['movie title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies[movies['movie title'] == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies[movies.index==index]['movie title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1
