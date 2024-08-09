import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsClassifier

# Load datasets
films = pd.read_csv(r"F:\Machine Learning\Movie_ Recommendation\movies_metadata.csv", on_bad_lines='skip')
link = pd.read_csv(r"F:\Machine Learning\Movie_ Recommendation\links.csv")

# Print initial overview
print(films.head(5))
print(films.shape)
print(films.columns)

# Select and rename key columns
films = films[['id', 'title', 'release_date', 'overview', 'tagline', 'poster_path', 'genres', 'imdb_id']]
link.rename(columns={"imdbId": "imdb_id"}, inplace=True)

# Data type conversions and string formatting
films['imdb_id'] = films['imdb_id'].astype(str)
link['imdb_id'] = 'tt0' + link['imdb_id'].astype(str)
films['id'] = films['id'].astype(str)
link['movieId'] = link['movieId'].astype(str)

# Display some unique identifiers for validation
print("Unique imdb in films:")
print(films['imdb_id'].unique()[:5])
print("Unique imdb in link:")
print(link['imdb_id'].unique()[:10])
print("Unique movieId in link:")
print(link['movieId'].unique()[:5])

# Merge film data with link data based on 'imdb'
merged_films = pd.merge(films, link[['movieId', 'imdb_id']], on='imdb_id', how='left')
print(merged_films.head(5))
print(merged_films.shape)
print(merged_films.columns)

# Load user ratings dataset
ratings = pd.read_csv(r"F:\Machine Learning\Movie_ Recommendation\ratings.csv")
print(ratings.head(5))

ratings['movieId'] = ratings['movieId'].astype(str)

# Display some unique movieIds for validation
print("Unique movieId in ratings:")
print(ratings['movieId'].unique()[:10])

# Count the number of unique users
print(ratings['userId'].unique().shape)

# Filter users with more than 200 ratings
active_users = ratings['userId'].value_counts()
top_users = active_users[active_users > 200].index

filtered_ratings = ratings[ratings['userId'].isin(top_users)]

# Display filtered ratings DataFrame
print(filtered_ratings.head(5))
print(filtered_ratings.shape)

# Merge user ratings with film data on 'movieId'
user_films = filtered_ratings.merge(merged_films, on="movieId", how='left')
print(user_films.head(5))
print(user_films.shape)

rating_counts = user_films.groupby('title')['rating'].count().reset_index()
print(rating_counts.head(5))
rating_counts.rename(columns={"rating": "rating_count"}, inplace=True)
final_ratings = user_films.merge(rating_counts, on='title')
print(final_ratings.head(5))

# Filter movies with at least 50 ratings and remove duplicates
final_ratings = final_ratings[final_ratings['rating_count'] >= 50]
print(final_ratings.head(5))
final_ratings.drop_duplicates(['userId', 'title'], inplace=True)

# Create a user-movie ratings pivot table
rating_pivot = final_ratings.pivot(columns='userId', index='title', values='rating')
print(rating_pivot)
rating_pivot.fillna(0, inplace=True)

# Convert the pivot table to a sparse matrix
sparse_ratings = csr_matrix(rating_pivot)

# Initialize and train the KNN model
knn = KNeighborsClassifier(metric='cosine', algorithm='brute')
knn.fit(sparse_ratings, np.arange(sparse_ratings.shape[0]))

# Save the model and pivot table
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

with open('rating_pivot.pkl', 'wb') as f:
    pickle.dump(rating_pivot, f)

# Function to retrieve movie recommendations using KNN
def recommend_movies(title, num_recs=5):
    if title not in rating_pivot.index:
        return f"Movie '{title}' not found in the dataset."
    
    title_idx = rating_pivot.index.get_loc(title)
    distances, indices = knn.kneighbors(sparse_ratings[title_idx], n_neighbors=num_recs+1)
    
    recommendations = [rating_pivot.index[i] for i in indices.flatten() if rating_pivot.index[i] != title]
    return recommendations[:num_recs]

# Example usage: Get recommendations based on a movie title
input_title = "The Lion King"  # replace with the actual movie title
recommendations = recommend_movies(input_title)
print(recommendations)

# Save the final films data to a CSV file
user_films.to_csv('final_films.csv')