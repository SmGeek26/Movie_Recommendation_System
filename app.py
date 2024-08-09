import streamlit as st
import pandas as pd
import pickle

# Load pre-trained KNN model and pivot table
with open('knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

with open('rating_pivot.pkl', 'rb') as f:
    rating_pivot = pickle.load(f)

# Load the film data with genres
film_cast = pd.read_csv('final_films.csv')

# Ensure the genres column is properly processed
film_cast['genres'] = film_cast['genres'].fillna('Unknown')

# Function to extract genre names from the JSON-like structure
def extract_genre_names(genre_list):
    try:
        genres = eval(genre_list)
        genre_names = [genre['name'] for genre in genres]
        return ', '.join(genre_names)
    except (SyntaxError, TypeError):
        return 'Unknown'

# Function to retrieve movie recommendations along with genres using KNN
def recommend_movies_with_genres(title, num_recs=5):
    if title not in rating_pivot.index:
        return f"Movie '{title}' not found in the dataset."
    
    title_idx = rating_pivot.index.get_loc(title)
    
    # Perform KNN search
    try:
        distances, indices = knn.kneighbors(rating_pivot.iloc[title_idx].values.reshape(1, -1), n_neighbors=num_recs+1)
    except Exception as e:
        st.write(f"Error during KNN computation: {e}")
        return []

    recommendations = []
    for i in indices.flatten():
        if rating_pivot.index[i] != title:
            movie_title = rating_pivot.index[i]
            movie_genre = film_cast[film_cast['title'] == movie_title]['genres'].values
            movie_genre_names = extract_genre_names(movie_genre[0]) if len(movie_genre) > 0 else 'Unknown'
            recommendations.append((movie_title, movie_genre_names))
    
    return recommendations[:num_recs]

# Streamlit app layout
st.title("Movie Recommender System")
st.write("Select a movie from the dropdown below to get similar movie recommendations with genres:")

# Movie selection dropdown
selected_movie = st.selectbox("Choose a movie:", rating_pivot.index)

# Number of recommendations slider
num_recs = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

# Display recommendations
if st.button("Recommend"):
    with st.spinner('Finding recommendations...'):
        recommendations = recommend_movies_with_genres(selected_movie, num_recs)
    
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write(f"Movies similar to '{selected_movie}':")
        for i, (rec_title, rec_genre) in enumerate(recommendations, start=1):
            st.write(f"{i}. *{rec_title}* - Genres: {rec_genre}")

# Run the app
if _name_ == '_main_':
    st.write("Use the options above to get movie recommendations.")