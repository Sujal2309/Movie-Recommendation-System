from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movies data
movies_data = pd.read_csv(r'C:\Users\VICTUS\Downloads\Movierecommendation sytem.zip')

# Preprocess the data for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                    movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculate cosine similarity
similarity = cosine_similarity(feature_vectors)

app = Flask(__name__)

# Define routes
@app.route('/')
def index():
    return render_template('idex.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    
    # Find close matches
    list_of_all_titles = movies_data['title'].tolist()
    close_matches = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if len(close_matches) > 0:
        close_match = close_matches[0]
        index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
        
        # Get similarity scores
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        
        # Sort movies by similarity
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        # Prepare recommended movies list
        recommended_movies = []
        for i, movie in enumerate(sorted_similar_movies):
            index = movie[0]
            title_from_index = movies_data.loc[index, 'title']
            if i < 30:  # Limiting to top 30 recommendations
                recommended_movies.append(title_from_index)
            else:
                break
        
        return render_template('idex.html', movie=close_match, recommended_movies=recommended_movies)
    else:
        return render_template('idex.html', error_message='No matches found for that movie.')

if __name__ == '__main__':
    app.run(debug=True)
