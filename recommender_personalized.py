import pandas as pd
from collections import defaultdict

# Genre list
genre_labels = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
                "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

# Load data
ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
columns = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_labels
movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", names=columns, usecols=range(24), engine="python")

# Kullanıcının en çok puan verdiği türleri hesapla
def get_user_genre_profile(user_id):
    user_ratings = ratings[ratings["user_id"] == user_id]
    genre_scores = defaultdict(float)

    for _, row in user_ratings.iterrows():
        movie = movies[movies["movie_id"] == row["movie_id"]]
        if movie.empty:
            continue
        for genre in genre_labels:
            if movie.iloc[0][genre] == 1:
                genre_scores[genre] += row["rating"]

    sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
    return [g[0] for g in sorted_genres[:3]]  # En çok sevilen 3 tür

# Belirtilen türlere göre öneri üret
def recommend_by_genre(genres):
    mask = movies[genres].sum(axis=1) > 0
    recommendations = movies[mask][["title"] + genres]
    return recommendations["title"].head(10)

# Ana uygulama
if __name__ == "__main__":
    print("🧑‍🤝‍🧑 Kişiselleştirilmiş Öneri (Kullanıcı Tercihli)")
    user_id = int(input("👤 Kullanıcı ID girin (1-943 arası): "))

    favorite_genres = get_user_genre_profile(user_id)
    if not favorite_genres:
        print("Kullanıcının tür profili bulunamadı.")
    else:
        print(f"🎯 Bu kullanıcının favori türleri: {', '.join(favorite_genres)}")
        print("🎬 Önerilen filmler:")
        recommended = recommend_by_genre(favorite_genres)
        for i, title in enumerate(recommended, 1):
            print(f"{i}. {title}")
