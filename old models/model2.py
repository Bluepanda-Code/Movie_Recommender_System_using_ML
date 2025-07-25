import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV


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
    return data, ratings, movies


# Train and evaluate SVD model
def train_model(data):
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD(n_factors=100, random_state=42)
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Save model
    with open('svd_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model, trainset, rmse, mae, cv_results


# Get top-N recommendations
def get_top_n_recommendations(model, trainset, ratings, movies, user_id, n=10):
    try:
        inner_uid = trainset.to_inner_uid(user_id)
    except ValueError:
        print(f"\nUser {user_id} is new or not in training data. Recommending popular movies.")
        popular = ratings.groupby('movieId').agg({'rating': 'mean', 'userId': 'count'}).reset_index()
        popular = popular[popular['userId'] > 20].sort_values('rating', ascending=False).head(n)
        top_n = popular.merge(movies[['movieId', 'title', 'genres']], on='movieId')
        top_n['predicted_rating'] = top_n['rating']
        top_n['reason'] = "Popular movie with high average rating"
        return top_n[['title', 'genres', 'predicted_rating', 'reason']]

    user_rated = [trainset.to_raw_iid(iid) for iid, _ in trainset.ur[inner_uid]]
    all_movie_ids = movies['movieId'].unique()

    # Filter out already-rated and unpopular movies
    popular_ids = ratings['movieId'].value_counts()
    popular_ids = popular_ids[popular_ids > 20].index
    filtered_ids = [mid for mid in all_movie_ids if mid not in user_rated and mid in popular_ids]

    predictions = [(mid, model.predict(user_id, mid).est) for mid in filtered_ids]
    predictions.sort(key=lambda x: x[1], reverse=True)

    top_n = pd.DataFrame(predictions[:n], columns=['movieId', 'predicted_rating'])
    top_n = top_n.merge(movies[['movieId', 'title', 'genres']], on='movieId')
    top_n['reason'] = "High predicted rating based on similar users"
    return top_n[['title', 'genres', 'predicted_rating', 'reason']]


# Optional: Hyperparameter tuning
def optimize_model(data):
    param_grid = {
        'n_factors': [50, 100],
        'lr_all': [0.002, 0.005],
        'reg_all': [0.02, 0.1]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)

    print("Best RMSE:", gs.best_score['rmse'])
    print("Best Parameters:", gs.best_params['rmse'])

    best_model = gs.best_estimator['rmse']
    best_model.fit(data.build_full_trainset())
    return best_model


# Main
if __name__ == "__main__":
    data, ratings, movies = load_data()

    # Uncomment if you want hyperparameter tuning
    # model = optimize_model(data)
    # trainset = data.build_full_trainset()

    model, trainset, rmse, mae, cv_results = train_model(data)

    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"CV RMSE: {cv_results['test_rmse'].mean():.4f}")
    print(f"CV MAE : {cv_results['test_mae'].mean():.4f}")

    # Get recommendations for user
    try:
        user_id = int(input("\nEnter user ID to get movie recommendations: "))
    except:
        user_id = 1

    recommendations = get_top_n_recommendations(model, trainset, ratings, movies, user_id)
    print(f"\nTop 10 movie recommendations for User {user_id}:\n")
    print(recommendations.to_string(index=False))
