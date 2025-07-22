import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

# Veri yükleme
ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", usecols=[0, 1], names=["movie_id", "title"], engine="python")

# User-movie matrisi oluştur
rating_matrix = ratings.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)

# Kullanıcı benzerlik matrisi (cosine similarity)
user_similarity = cosine_similarity(rating_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

# Öneri fonksiyonu
def recommend(user_id, top_n=10):
    if user_id not in rating_matrix.index:
        print("Geçersiz kullanıcı ID")
        return []

    # Benzer kullanıcıları al
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]

    # Kullanıcının izlemediği filmleri topla
    user_ratings = rating_matrix.loc[user_id]
    unseen_movies = user_ratings[user_ratings == 0].index

    # Öneri skorlarını hesapla
    movie_scores = {}
    for movie_id in unseen_movies:
        score = 0
        total_sim = 0
        for sim_user_id, sim_score in similar_users.items():
            sim_user_rating = rating_matrix.loc[sim_user_id, movie_id]
            if sim_user_rating > 0:
                score += sim_score * sim_user_rating
                total_sim += sim_score
        if total_sim > 0:
            movie_scores[movie_id] = score / total_sim

    # En yüksek skorlu filmler
    top_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_movie_ids = [m[0] for m in top_movies]
    return movies[movies["movie_id"].isin(top_movie_ids)]["title"].values

# Kullanıcıdan input al
if __name__ == "__main__":
    print("👥 Kullanıcı Tabanlı Öneri Sistemi (Collaborative Filtering)")
    try:
        user_id = int(input("Kullanıcı ID girin (1-943): "))
        recommended = recommend(user_id)
        if len(recommended) == 0:
            print("Öneri bulunamadı.")
        else:
            print("🔮 Sizin için önerilen filmler:")
            for i, title in enumerate(recommended, 1):
                print(f"{i}. {title}")
    except:
        print("Geçersiz giriş.")
