import pandas as pd
from datetime import datetime

# Genre list
genre_labels = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
                "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

# Load movie data
columns = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_labels
movies = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", names=columns, usecols=range(24), engine="python")

# Yıl filtresine göre film öner
def recommend_by_year(start_year=None, end_year=None):
    filtered = movies.copy()
    filtered["year"] = pd.to_datetime(filtered["release_date"], errors="coerce").dt.year

    if start_year:
        filtered = filtered[filtered["year"] >= start_year]
    if end_year:
        filtered = filtered[filtered["year"] <= end_year]

    return filtered["title"].dropna().head(10)

# Ana uygulama
if __name__ == "__main__":
    print("📅 Zaman Filtresiyle Film Önerici")
    try:
        start = int(input("Başlangıç yılı (örn: 1995): "))
    except:
        start = None
    try:
        end = int(input("Bitiş yılı (örn: 2000): "))
    except:
        end = None

    print(f"🎬 {start if start else ''} - {end if end else ''} arası çıkan önerilen filmler:")
    recommendations = recommend_by_year(start, end)
    for i, title in enumerate(recommendations, 1):
        print(f"{i}. {title}")
