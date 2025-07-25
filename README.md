# 🎬 Movie Recommender System

![Movie Recommender Banner](https://github.com/Bluepanda-Code/Movie_Recommender_System_using_ML/blob/master/git_images/banner.png)

A Flask-based movie recommender system that combines **collaborative filtering** (SVD) and **content-based filtering** (TF-IDF on movie tags and genres) to provide personalized movie suggestions. Built using Python, Flask, and scikit-learn, with interactive visualizations and TMDb API integration. 🚀

---

## ✨ Features

- 🔁 **Hybrid Recommendations**: Combines SVD + TF-IDF for personalized results.
- 🎯 **Content-Based Filtering**: Recommends movies similar to your favorites.
- 🌐 **Web Interface**: Search via user ID or movie title with poster + IMDb/TMDb links.
- 📊 **Visualizations**: Rating and genre distribution charts.
- 🎬 **TMDb API Integration**: Fetches real-time movie posters and metadata.

---

## 📂 Dataset

The system uses the **MovieLens dataset**, including:

- `movies2.csv`: Movie metadata (movieId, title, genres)
- `ratings2.csv`: User ratings (userId, movieId, rating)
- `links.csv`: Mapping of movieId to IMDb and TMDb
- `tags.csv`: User-generated tags per movie

> ⚠️ `movies2.csv` and `ratings2.csv` use a **semicolon (;)** as the delimiter. Files are stored in the `data/` folder.

---

## ⚙️ Setup

### ✅ Prerequisites

- Suggested Python Version v3.10 (the `scikit-surprise` library doesn't work with python latest versions like 3.13 )
- Git
- TMDb API key ([Register here](https://www.themoviedb.org/))
  Go to the tmdb link, and click on Join for signup. Now go to `profile->Settings->API->personal use`
  ⚠️In case it asks for url, put this as url - `http://127.0.0.1:5000`

---

### 🧰 Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Bluepanda-Code/Movie_Recommender_System_using_ML.git
   cd Movie_Recommender_System_using_ML
   ```
2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   ```

   * On Windows:

     ```bash
     .\venv\Scripts\activate
     ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up TMDb API Key**

   * Open `main.py` and replace `'your_tmdb_api_key'` with your actual key.

5. **Prepare Static Folder for storing visualization images**

   ```bash
   mkdir static
   ```

---

## 🚀 Running the App

Launch the Flask server:

```bash
python main.py
```

Access it in your browser at:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📊 Visualizations

### Rating Distribution

![Rating Distribution](https://github.com/Bluepanda-Code/Movie_Recommender_System_using_ML/blob/master/git_images/rating_distribution.png)

### Genre Distribution

![Genre Distribution](https://github.com/Bluepanda-Code/Movie_Recommender_System_using_ML/blob/master/git_images/genre_distribution.png)

---

## 🖼️ Screenshots


### Web-Input Interface

![Web-Input Interface](https://github.com/Bluepanda-Code/Movie_Recommender_System_using_ML/blob/master/git_images/web_interface.png)

### User-Based Recommendations

![User-Based Recommendations](https://github.com/Bluepanda-Code/Movie_Recommender_System_using_ML/blob/master/git_images/user_based.png)

### Content-Based Recommendations

![Content-Based Recommendations](https://github.com/Bluepanda-Code/Movie_Recommender_System_using_ML/blob/master/git_images/content_based.png)

---

## 🤝 Contributing

Contributions are welcome!

```bash
git checkout -b feature/YourFeature
# Make your changes
git commit -m "Add Your Feature"
git push origin feature/YourFeature
```

Then open a **pull request**.

---

## 🔮 Future Improvements

* 💄 Enhance UI styling (e.g. use `app_logo.jpg`)
* ⭐ Show user-rated movies alongside recommendations
* 🧹 Improve tag preprocessing for content filtering
* 🗄 Add a database (e.g., SQLite/PostgreSQL) for scalability
* 🗳 Add user feedback on recommendations
* 📈 More visualizations (interactive plots, dashboards)

---

## 📧 Contact

For questions or suggestions, open a GitHub issue or reach out:
📮 **[namanvish072@gmail.com](mailto:namanvish072@gmail.com)**

---

🎉 Happy movie watching! 🍿
