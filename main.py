from flask import Flask, redirect, url_for, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import pickle

# Load pretrained Model
with open('Movie_recommend.pkl', 'rb') as file:
   vectorizer,similarity,movies = pickle.load(file)
   
app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
   
   # movie_name = input(' Enter your favourite movie name : ')
   movie_name = name
   list_of_all_titles = movies['movie title'].tolist()

   find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

   close_match = find_close_match[0]

   index_of_the_movie = movies[movies['movie title'] == close_match]['index'].values[0]

   similarity_score = list(enumerate(similarity[index_of_the_movie]))

   sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
   print('Movies suggested for you : \n')
   
   i = 0
   movieList = []
   for movie in sorted_similar_movies:
      index = movie[0]
      title_from_index = movies[movies.index==index]['movie title'].values[0]
      
      if (i<10):
         print(i+1, '.',title_from_index)
         movieList.append(title_from_index)
         i+=1
   return render_template("index.html", Movies = movieList ) 

@app.route('/search',methods = ['POST', 'GET'])
def search():
   if request.method == 'POST':
      movie_name = request.form['mname']
      return redirect(url_for('success', name = movie_name))
   else:
      movie_name = request.args.get('mname')
      return redirect(url_for('success', name = movie_name)) 


@app.route('/')
def index():
   return render_template("index.html")


if __name__ == "__main__":
   app.run(debug=True, port = 7500)


# return render_template("result.html",result = result)