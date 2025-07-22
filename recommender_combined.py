import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Genre list
genre_labels = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
                "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

# Load datasets
ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
columns = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_labels
movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", names=columns, usecols=range(24), engine="python")

# Genre string
def get_genre_string(row):
    return ' '.join([genre for genre in genre_labels if row[genre] == 1])
movies["genres"] = movies.apply(get_genre_string, axis=1)

# Hybrid content string
movies["content"] = movies["title"] + " " + movies["genres"]

# TF-IDF: Title-based
def tfidf_title_model():
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["title"])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# TF-IDF: Genre-based
def tfidf_genre_model():
    tfidf = TfidfVectorizer(token_pattern=r'[^|\s]+')
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# TF-IDF: Hybrid (title + genre)
def tfidf_hybrid_model():
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["content"])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# Title index
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# Recommend function
def recommend(title, similarity_matrix):
    if title not in indices:
        print(f"'{title}' bulunamadı.")
        return []
    idx = indices[title]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices]

# Main interface
if __name__ == "__main__":
    print("🎬 Gelişmiş Netflix Film Öneri Sistemi")
    print("1 - Başlık tabanlı öneri")
    print("2 - Tür tabanlı öneri")
    print("3 - Hibrit (Başlık + Tür) öneri")
    choice = input("Lütfen öneri tipi seçin (1/2/3): ")

    title = input("📥 Film ismini girin (örn: Star Wars (1977)): ")

    if choice == "1":
        sim = tfidf_title_model()
    elif choice == "2":
        sim = tfidf_genre_model()
    elif choice == "3":
        sim = tfidf_hybrid_model()
    else:
        print("Geçersiz seçim.")
        exit()

    print(f"🔮 '{title}' için önerilen filmler:")
    recommendations = recommend(title, sim)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
