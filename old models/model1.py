import pandas as pd
import matplotlib.pyplot as plt
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
import os
import csv

# Load and preprocess data
def load_data(movies_path='data/movies2.csv', ratings_path='data/ratings2.csv'):
    print(f"Looking for movies2.csv at: {os.path.abspath(movies_path)}")
    print(f"Looking for ratings2.csv at: {os.path.abspath(ratings_path)}")
    movies = pd.read_csv(movies_path, delimiter=';', encoding='latin1')
    ratings = pd.read_csv(ratings_path, delimiter=';', encoding='latin1')

    if ratings[['userId', 'movieId', 'rating']].isnull().any().any():
        print("Warning: Missing values in ratings. Dropping rows.")
        ratings = ratings.dropna()
    
    # Plot rating distribution
    plt.figure(figsize=(8, 6))
    ratings['rating'].hist(bins=10, edgecolor='black', color='#4C78A8')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('rating_distribution.png')
    plt.close()
    
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    return data, movies

# Train and evaluate SVD model
def train_model(data):
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD(n_factors=100, random_state=42)
    model.fit(trainset)
    
    predictions = model.test(testset)
    from surprise import accuracy
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    
    cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return model, trainset, rmse, mae, cv_results

# Get top-N recommendations
def get_top_n_recommendations(model, trainset, movies, user_id, n=10):
    all_movie_ids = movies['movieId'].unique()
    user_rated = [trainset.to_raw_iid(iid) for iid, _ in trainset.ur[trainset.to_inner_uid(user_id)]]
    
    predictions = [(mid, model.predict(user_id, mid).est) for mid in all_movie_ids if mid not in user_rated]
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    top_n = pd.DataFrame(predictions[:n], columns=['movieId', 'predicted_rating'])
    top_n = top_n.merge(movies[['movieId', 'title', 'genres']], on='movieId')
    return top_n

# Main
if __name__ == "__main__":
    data, movies = load_data('data/movies2.csv', 'data/ratings2.csv')
    model, trainset, rmse, mae, cv_results = train_model(data)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"CV RMSE: {cv_results['test_rmse'].mean():.4f}")
    
    user_id = 1
    recommendations = get_top_n_recommendations(model, trainset, movies, user_id)
    print(f"\nTop 10 recommendations for user {user_id}:")
    print(recommendations[['title', 'genres', 'predicted_rating']])