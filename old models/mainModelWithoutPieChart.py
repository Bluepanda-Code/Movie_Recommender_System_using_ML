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
tmdb.api_key = os.getenv('TMDB_API_KEY', '64b1c927bbba4616649781093acc0ffd') #<------ Get a tmdb api key and replace with mine
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

    movies = pd.read_csv(movies_path, delimiter=';', encoding='latin1') #<----- The  dataset is in format with separation as ";" hence (delimeter=';').
    ratings = pd.read_csv(ratings_path, delimiter=';', encoding='latin1')
    links = pd.read_csv(links_path, encoding='latin1') #<----- BY default if not mentioned the delimeter will be ","
    tags = pd.read_csv(tags_path, encoding='latin1')

    if ratings[['userId', 'movieId', 'rating']].isnull().any().any():
        print("Warning: Missing values in ratings. Dropping rows.")
        ratings = ratings.dropna()

    if tags[['movieId', 'tag']].isnull().any().any():
        print("Warning: Missing values in tags. Dropping rows.")
        tags = tags.dropna(subset=['movieId', 'tag'])

    # Filter tags to only include movieIds present in movies2.csv
    valid_movie_ids = set(movies['movieId'])
    tags = tags[tags['movieId'].isin(valid_movie_ids)]
    print(f"Unique movies in tags.csv after filtering: {len(tags['movieId'].unique())}")
    print(f"Unique movies in movies2.csv: {len(movies['movieId'].unique())}")

    # Plot rating distribution
    plt.figure(figsize=(8, 6))
    ratings['rating'].hist(bins=10, edgecolor='black', color='#4C78A8')
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('static/rating_distribution.png')
    plt.close()

    # Preprocess tags for content-based filtering
    tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    tags_grouped = tags_grouped.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
    # Fallback to genres if tags are missing
    tags_grouped['tag'] = tags_grouped.apply(lambda x: x['genres'] if pd.isna(x['tag']) or x['tag'] == '' else x['tag'], axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(tags_grouped['tag'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    movie_indices = pd.Series(tags_grouped.index, index=tags_grouped['movieId'])

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    return data, ratings, movies, links, tags_grouped, cosine_sim, movie_indices

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
def get_content_recommendations(movie_id, tags_grouped, cosine_sim, movie_indices, movies, links, ratings, n=10):
    if movie_id not in movie_indices:
        print(f"Movie ID {movie_id} not found in tags data. Recommending popular movies.")
        popular = ratings.groupby('movieId').agg({'rating': 'mean', 'userId': 'count'}).reset_index()
        popular = popular[popular['userId'] > 20].sort_values('rating', ascending=False).sample(n, random_state=42)  # Randomize for diversity
        top_n = popular.merge(movies[['movieId', 'title', 'genres']], on='movieId')
        top_n = top_n.merge(links[['movieId', 'imdbId', 'tmdbId']], on='movieId', how='left')
        top_n['predicted_rating'] = top_n['rating']
        top_n['reason'] = "Popular movie with high average rating"
        top_n['imdb_link'] = top_n['imdbId'].apply(lambda x: f"https://www.imdb.com/title/tt{x:07d}")
        top_n['details'] = top_n['tmdbId'].apply(get_movie_details)
        print(f"Popular recommendations: {top_n[['movieId', 'title']].to_dict('records')}")
        return top_n[['title', 'genres', 'predicted_rating', 'imdb_link', 'tmdbId', 'details', 'reason']]

    idx = movie_indices[movie_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices_top = [i[0] for i in sim_scores]
    top_n = tags_grouped.iloc[movie_indices_top][['movieId', 'title', 'genres']]
    top_n = top_n.merge(links[['movieId', 'imdbId', 'tmdbId']], on='movieId', how='left')
    top_n['predicted_rating'] = top_n['movieId'].map(ratings.groupby('movieId')['rating'].mean()).fillna(3.7)
    top_n['reason'] = f"Similar to movie ID {movie_id} based on tags/genres"
    top_n['imdb_link'] = top_n['imdbId'].apply(lambda x: f"https://www.imdb.com/title/tt{x:07d}")
    top_n['details'] = top_n['tmdbId'].apply(get_movie_details)
    print(f"Content-based recommendations for movie {movie_id}: {top_n[['movieId', 'title']].to_dict('records')}")
    return top_n[['title', 'genres', 'predicted_rating', 'imdb_link', 'tmdbId', 'details', 'reason']]

# Get hybrid recommendations (SVD + content-based)
def get_hybrid_recommendations(model, trainset, ratings, movies, links, tags_grouped, cosine_sim, movie_indices, user_id, n=10):
    try:
        inner_uid = trainset.to_inner_uid(user_id)
        # Get SVD predictions
        user_rated = [trainset.to_raw_iid(iid) for iid, _ in trainset.ur[inner_uid]]
        all_movie_ids = movies['movieId'].unique()
        popular_ids = ratings['movieId'].value_counts()
        popular_ids = popular_ids[popular_ids > 20].index
        filtered_ids = [mid for mid in all_movie_ids if mid not in user_rated and mid in popular_ids]
        svd_predictions = [(mid, model.predict(user_id, mid).est) for mid in filtered_ids]
        top_svd = pd.DataFrame(svd_predictions, columns=['movieId', 'svd_score'])
        print(f"User {user_id} SVD scores (top 5): {top_svd.head().to_dict('records')}")

        # Normalize SVD scores to 0-1
        max_svd = top_svd['svd_score'].max()
        min_svd = top_svd['svd_score'].min()
        if max_svd != min_svd:
            top_svd['svd_score'] = (top_svd['svd_score'] - min_svd) / (max_svd - min_svd) * 5  # Scale to 0-5
        else:
            top_svd['svd_score'] = 5.0

        # Get content-based scores based on user's highest-rated movie
        user_ratings = ratings[ratings['userId'] == user_id][['movieId', 'rating']]
        if not user_ratings.empty:
            top_rated_movie = user_ratings.sort_values('rating', ascending=False).iloc[0]['movieId']
            print(f"User {user_id} top-rated movie: {top_rated_movie}")
            content_recs = get_content_recommendations(top_rated_movie, tags_grouped, cosine_sim, movie_indices, movies, links, ratings, n=n*2)
        else:
            # Fallback to a random popular movie
            popular_movie = ratings[ratings['movieId'].isin(popular_ids)]['movieId'].sample(random_state=user_id).iloc[0]
            print(f"User {user_id} has no ratings, using movie {popular_movie} for content-based.")
            content_recs = get_content_recommendations(popular_movie, tags_grouped, cosine_sim, movie_indices, movies, links, ratings, n=n*2)

        # Ensure content_recs has movieId and content_score
        content_recs = content_recs.rename(columns={'predicted_rating': 'content_score'})
        if content_recs.empty or 'movieId' not in content_recs.columns:
            print(f"Content-based recommendations empty or missing movieId for user {user_id}. Using SVD only.")
            content_recs = pd.DataFrame({'movieId': top_svd['movieId'], 'content_score': 0.0})

        # Normalize content-based scores to 0-5
        if not content_recs.empty:
            max_content = content_recs['content_score'].max()
            min_content = content_recs['content_score'].min()
            if max_content != min_content:
                content_recs['content_score'] = (content_recs['content_score'] - min_content) / (max_content - min_content) * 5
            else:
                content_recs['content_score'] = 5.0
        print(f"Content-based scores (top 5): {content_recs.head().to_dict('records')}")

        # Combine SVD and content-based scores
        top_n = top_svd.merge(content_recs[['movieId', 'content_score']], on='movieId', how='left')
        top_n['content_score'] = top_n['content_score'].fillna(0.0)
        top_n['hybrid_score'] = 0.7 * top_svd['svd_score'] + 0.3 * top_n['content_score']
        top_n = top_n.sort_values('hybrid_score', ascending=False).head(n)
        print(f"Hybrid scores (top 5): {top_n[['movieId', 'hybrid_score']].to_dict('records')}")

        top_n = top_n.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
        top_n = top_n.merge(links[['movieId', 'imdbId', 'tmdbId']], on='movieId', how='left')
        top_n['reason'] = "Hybrid: SVD prediction + tag/genres similarity"
        top_n['imdb_link'] = top_n['imdbId'].apply(lambda x: f"https://www.imdb.com/title/tt{x:07d}")
        top_n['details'] = top_n['tmdbId'].apply(get_movie_details)
        return top_n[['title', 'genres', 'hybrid_score', 'imdb_link', 'tmdbId', 'details', 'reason']]
    except ValueError:
        print(f"User {user_id} is new. Recommending content-based for a random popular movie.")
        popular_movie = ratings[ratings['movieId'].isin(ratings['movieId'].value_counts().index[ratings['movieId'].value_counts() > 20])]['movieId'].sample(random_state=user_id).iloc[0]
        return get_content_recommendations(popular_movie, tags_grouped, cosine_sim, movie_indices, movies, links, ratings, n=n)

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        movie_id = request.form.get('movie_id')
        
        if user_id:
            try:
                user_id = int(user_id)
                recommendations = get_hybrid_recommendations(model, trainset, ratings, movies, links, tags_grouped, cosine_sim, movie_indices, user_id)
            except Exception as e:
                print(f"Error for user {user_id}: {e}")
                user_id = 1
                recommendations = get_hybrid_recommendations(model, trainset, ratings, movies, links, tags_grouped, cosine_sim, movie_indices, user_id)
        elif movie_id:
            try:
                movie_id = int(movie_id)
                recommendations = get_content_recommendations(movie_id, tags_grouped, cosine_sim, movie_indices, movies, links, ratings)
            except Exception as e:
                print(f"Error for movie {movie_id}: {e}")
                movie_id = 1
                recommendations = get_content_recommendations(movie_id, tags_grouped, cosine_sim, movie_indices, movies, links, ratings)
        else:
            user_id = 1
            recommendations = get_hybrid_recommendations(model, trainset, ratings, movies, links, tags_grouped, cosine_sim, movie_indices, user_id)

        recommendations['poster_path'] = recommendations['details'].apply(lambda x: x['poster_path'])
        recommendations['overview'] = recommendations['details'].apply(lambda x: x['overview'])
        recommendations['release_date'] = recommendations['details'].apply(lambda x: x['release_date'])

        return render_template_string("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Movie Recommender</title>
                <link rel="icon" type="image/x-icon" href="/static/app_logo.jpg">
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #4C78A8; color: white; }
                    tr:nth-child(even) { background-color: #f2f2f2; }
                    h1, h2 { color: #4C78A8; }
                    img { max-width: 100px; height: auto; }
                    select { width: 300px; padding: 5px; }
                </style>
            </head>
            <body>
                <h1>Movie Recommender System</h1>
                <form method="post">
                    <label for="user_id">Enter User ID for Hybrid Recommendations:</label>
                    <input type="number" id="user_id" name="user_id" value="{{ user_id if user_id else '' }}">
                    <br><br>
                    <label for="movie_id">Or Select a Movie for Content-Based Recommendations:</label>
                    <select id="movie_id" name="movie_id">
                        <option value="">Select a movie</option>
                        {% for movie in movies %}
                        <option value="{{ movie.movieId }}">{{ movie.title }}</option>
                        {% endfor %}
                    </select>
                    <br><br>
                    <button type="submit">Get Recommendations</button>
                </form>
                <h2>Recommendations {% if user_id %}for User {{ user_id }}{% elif movie_id %}Similar to Movie ID {{ movie_id }}{% endif %}</h2>
                <table>
                    <tr>
                        <th>Poster</th>
                        <th>Title</th>
                        <th>Genres</th>
                        <th>Score</th>
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
                        <td>{{ rec.hybrid_score | round(2) if rec.hybrid_score is defined else rec.predicted_rating | round(2) }}</td>
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
        """, recommendations=recommendations.to_dict('records'), user_id=user_id, movie_id=movie_id, movies=movies[['movieId', 'title']].to_dict('records'))
    
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Movie Recommender</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #4C78A8; }
                select { width: 300px; padding: 5px; }
            </style>
        </head>
        <body>
            <h1>Movie Recommender System</h1>
            <form method="post">
                <label for="user_id">Enter User ID for Hybrid Recommendations:</label>
                <input type="number" id="user_id" name="user_id">
                <br><br>
                <label for="movie_id">Or Select a Movie for Content-Based Recommendations:</label>
                <select id="movie_id" name="movie_id">
                    <option value="">Select a movie</option>
                    {% for movie in movies %}
                    <option value="{{ movie.movieId }}">{{ movie.title }}</option>
                    {% endfor %}
                </select>
                <br><br>
                <button type="submit">Get Recommendations</button>
            </form>
        </body>
        </html>
    """, movies=movies[['movieId', 'title']].to_dict('records'))

# Load data and train model at startup
data, ratings, movies, links, tags_grouped, cosine_sim, movie_indices = load_data()
model, trainset, rmse, mae, cv_results = train_model(data)

if __name__ == "__main__":
    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"CV RMSE: {cv_results['test_rmse'].mean():.4f}")
    print(f"CV MAE : {cv_results['test_mae'].mean():.4f}")
    app.run(debug=True)