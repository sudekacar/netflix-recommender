import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Genre list as per MovieLens 100K
genre_labels = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
                "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

# Load movies with genres
columns = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_labels
movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", names=columns, usecols=range(24), engine="python")

# Create genre string (e.g., "Action|Thriller")
def get_genre_string(row):
    return '|'.join([genre for genre in genre_labels if row[genre] == 1])

movies["genres"] = movies.apply(get_genre_string, axis=1)

# TF-IDF on genres
tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Title to index mapping
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# Recommendation function
def recommend(title, cosine_sim=cosine_sim):
    if title not in indices:
        print(f"'{title}' bulunamadı.")
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices]

# Interactive input
if __name__ == "__main__":
    print("🎬 Netflix Tür Bazlı Film Önerici")
    title = input("📥 Film ismini girin (örn: Star Wars (1977)): ")
    print(f"🔮 '{title}' için önerilen tür bazlı filmler:")
    recommendations = recommend(title)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
