import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate
from flask import Flask, request, render_template_string
from tmdbv3api import TMDb, Movie
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Initialize TMDb API
tmdb = TMDb()
tmdb.api_key = os.getenv('TMDB_API_KEY', '64b1c927bbba4616649781093acc0ffd')  # Replace with your actual key if not using env variable
tmdb.language = 'en'
movie_api = Movie()

# Function to fetch movie details from TMDb
def get_movie_details(tmdb_id):
    try:
        movie = movie_api.details(tmdb_id)
        return {
            'poster_path': f"https://image.tmdb.org/t/p/w200{movie.poster_path}" if movie.poster_path else None,
            'overview': movie.overview or "No summary available",
            'release_date': movie.release_date or "Unknown"
        }
    except Exception as e:
        print(f"Error fetching TMDb data for ID {tmdb_id}: {e}")
        return {'poster_path': None, 'overview': "No summary available", 'release_date': "Unknown"}

# Load and preprocess data
def load_data(movies_path='data/movies2.csv', ratings_path='data/ratings2.csv', links_path='data/links.csv', tags_path='data/tags.csv'):
    print(f"Looking for movies2.csv at: {os.path.abspath(movies_path)}")
    print(f"Looking for ratings2.csv at: {os.path.abspath(ratings_path)}")
    print(f"Looking for links.csv at: {os.path.abspath(links_path)}")
    print(f"Looking for tags.csv at: {os.path.abspath(tags_path)}")

    movies = pd.read_csv(movies_path, delimiter=';', encoding='latin1')
    ratings = pd.read_csv(ratings_path, delimiter=';', encoding='latin1')
    links = pd.read_csv(links_path, encoding='latin1')
    tags = pd.read_csv(tags_path, encoding='latin1')

    if ratings[['userId', 'movieId', 'rating']].isnull().any().any():
        print("Warning: Missing values in ratings. Dropping rows.")
        ratings = ratings.dropna()

    if tags[['movieId', 'tag']].isnull().any().any():
        print("Warning: Missing values in tags. Dropping rows.")
        tags = tags.dropna(subset=['movieId', 'tag'])

    # Plot rating distribution
    plt.figure(figsize=(8, 6))
    ratings['rating'].hist(bins=10, edgecolor='black', color='#4C78A8')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('static/rating_distribution.png')
    plt.close()

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    
    # Prepare tags for content-based filtering
    tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    tags_grouped = tags_grouped.merge(movies[['movieId', 'title']], on='movieId', how='left')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(tags_grouped['tag'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return data, ratings, movies, links, tags_grouped, cosine_sim

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

# Get content-based recommendations
def get_content_based_recommendations(user_id, ratings, movies, tags_grouped, cosine_sim, n=10):
    # Get movies rated highly (rating >= 4) by the user
    user_ratings = ratings[ratings['userId'] == user_id][['movieId', 'rating']]
    high_rated = user_ratings[user_ratings['rating'] >= 4]['movieId'].tolist()
    
    if not high_rated:
        print(f"No high-rated movies for user {user_id}. Recommending popular movies.")
        popular = ratings.groupby('movieId').agg({'rating': 'mean', 'userId': 'count'}).reset_index()
        popular = popular[popular['userId'] > 20].sort_values('rating', ascending=False).head(n)
        return popular.merge(movies[['movieId', 'title', 'genres']], on='movieId')

    # Find similar movies based on tags
    sim_scores = []
    for movie_id in high_rated:
        if movie_id in tags_grouped['movieId'].values:
            idx = tags_grouped[tags_grouped['movieId'] == movie_id].index[0]
            sim_scores.extend(list(enumerate(cosine_sim[idx])))
    
    if not sim_scores:
        print(f"No tag data for user {user_id}'s high-rated movies. Recommending popular movies.")
        popular = ratings.groupby('movieId').agg({'rating': 'mean', 'userId': 'count'}).reset_index()
        popular = popular[popular['userId'] > 20].sort_values('rating', ascending=False).head(n)
        return popular.merge(movies[['movieId', 'title', 'genres']], on='movieId')

    # Sort and select top similar movies
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[:n] if tags_grouped.iloc[i[0]]['movieId'] not in high_rated]
    top_n = tags_grouped.iloc[top_indices][['movieId', 'title']].merge(movies[['movieId', 'genres']], on='movieId')
    top_n['predicted_rating'] = 4.0  # Placeholder rating for content-based
    top_n['reason'] = "Similar tags to your highly rated movies"
    return top_n[['title', 'genres', 'predicted_rating', 'reason']]

# Get hybrid recommendations
def get_hybrid_recommendations(model, trainset, ratings, movies, links, tags_grouped, cosine_sim, user_id, n=10):
    # Get SVD-based recommendations
    try:
        inner_uid = trainset.to_inner_uid(user_id)
        user_rated = [trainset.to_raw_iid(iid) for iid, _ in trainset.ur[inner_uid]]
        all_movie_ids = movies['movieId'].unique()
        popular_ids = ratings['movieId'].value_counts()
        popular_ids = popular_ids[popular_ids > 20].index
        filtered_ids = [mid for mid in all_movie_ids if mid not in user_rated and mid in popular_ids]
        predictions = [(mid, model.predict(user_id, mid).est) for mid in filtered_ids]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n_svd = pd.DataFrame(predictions[:n//2], columns=['movieId', 'predicted_rating'])
        top_n_svd = top_n_svd.merge(movies[['movieId', 'title', 'genres']], on='movieId')
        top_n_svd['reason'] = "High predicted rating based on similar users"
    except ValueError:
        print(f"User {user_id} is new or not in training data. Using popular movies for SVD part.")
        popular = ratings.groupby('movieId').agg({'rating': 'mean', 'userId': 'count'}).reset_index()
        popular = popular[popular['userId'] > 20].sort_values('rating', ascending=False).head(n//2)
        top_n_svd = popular.merge(movies[['movieId', 'title', 'genres']], on='movieId')
        top_n_svd['predicted_rating'] = top_n_svd['rating']
        top_n_svd['reason'] = "Popular movie with high average rating"

    # Get content-based recommendations
    top_n_content = get_content_based_recommendations(user_id, ratings, movies, tags_grouped, cosine_sim, n=n//2)

    # Combine recommendations
    top_n = pd.concat([top_n_svd, top_n_content], ignore_index=True)
    top_n = top_n.merge(links[['movieId', 'imdbId', 'tmdbId']], on='movieId', how='left')
    top_n['imdb_link'] = top_n['imdbId'].apply(lambda x: f"https://www.imdb.com/title/tt{x:07d}" if pd.notnull(x) else "#")
    top_n['details'] = top_n['tmdbId'].apply(lambda x: get_movie_details(x) if pd.notnull(x) else {'poster_path': None, 'overview': "No summary available", 'release_date': "Unknown"})
    top_n['poster_path'] = top_n['details'].apply(lambda x: x['poster_path'])
    top_n['overview'] = top_n['details'].apply(lambda x: x['overview'])
    top_n['release_date'] = top_n['details'].apply(lambda x: x['release_date'])
    return top_n[['title', 'genres', 'predicted_rating', 'imdb_link', 'tmdbId', 'poster_path', 'overview', 'release_date', 'reason']]

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        try:
            user_id = int(request.form['user_id'])
        except:
            user_id = 1
        recommendations = get_hybrid_recommendations(model, trainset, ratings, movies, links, tags_grouped, cosine_sim, user_id)
        return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Movie Recommender</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #4C78A8; color: white; }
                    tr:nth-child(even) { background-color: #f2f2f2; }
                    h1, h2 { color: #4C78A8; }
                    img { max-width: 100px; height: auto; }
                </style>
            </head>
            <body>
                <h1>Movie Recommender System</h1>
                <form method="post">
                    <label for="user_id">Enter User ID:</label>
                    <input type="number" id="user_id" name="user_id" value="{{ user_id }}" required>
                    <button type="submit">Get Recommendations</button>
                </form>
                <h2>Recommendations for User {{ user_id }}</h2>
                <table>
                    <tr>
                        <th>Poster</th>
                        <th>Title</th>
                        <th>Genres</th>
                        <th>Predicted Rating</th>
                        <th>IMDb Link</th>
                        <th>Summary</th>
                        <th>Release Date</th>
                        <th>Reason</th>
                    </tr>
                    {% for rec in recommendations %}
                    <tr>
                        <td>{% if rec.poster_path %}<img src="{{ rec.poster_path }}" alt="{{ rec.title }}">{% else %}No Image{% endif %}</td>
                        <td>{{ rec.title }}</td>
                        <td>{{ rec.genres }}</td>
                        <td>{{ rec.predicted_rating | round(2) }}</td>
                        <td><a href="{{ rec.imdb_link }}" target="_blank">IMDb</a></td>
                        <td>{{ rec.overview }}</td>
                        <td>{{ rec.release_date }}</td>
                        <td>{{ rec.reason }}</td>
                    </tr>
                    {% endfor %}
                </table>
                <h2>Rating Distribution</h2>
                <img src="/static/rating_distribution.png" alt="Rating Distribution">
            </body>
            </html>
        """, recommendations=recommendations.to_dict('records'), user_id=user_id)
    
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Movie Recommender</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #4C78A8; }
            </style>
        </head>
        <body>
            <h1>Movie Recommender System</h1>
            <form method="post">
                <label for="user_id">Enter User ID:</label>
                <input type="number" id="user_id" name="user_id" required>
                <button type="submit">Get Recommendations</button>
            </form>
        </body>
        </html>
    """)

# Load data and train model at startup
data, ratings, movies, links, tags_grouped, cosine_sim = load_data()
model, trainset, rmse, mae, cv_results = train_model(data)

if __name__ == "__main__":
    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"CV RMSE: {cv_results['test_rmse'].mean():.4f}")
    print(f"CV MAE : {cv_results['test_mae'].mean():.4f}")
    app.run(debug=True)