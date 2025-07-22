import streamlit as st
import pandas as pd
from recommender_combined import tfidf_title_model, tfidf_genre_model, tfidf_hybrid_model, recommend as content_recommend
from recommender_personalized import get_user_genre_profile, recommend_by_genre
from recommender_time_filtered import recommend_by_year
from recommender_collaborative import recommend as collab_recommend

st.title("🎬 Netflix Film Öneri Uygulaması")

option = st.sidebar.selectbox("Öneri Yöntemi Seçin", [
    "İçerik Tabanlı (Başlık)",
    "İçerik Tabanlı (Tür)",
    "İçerik Tabanlı (Hibrit)",
    "Kişiselleştirilmiş (Genre + Rating)",
    "Zaman Filtresiyle Öneri",
    "Kullanıcı Bazlı (Collaborative Filtering)"
])

if option.startswith("İçerik Tabanlı"):
    title = st.text_input("Film adı girin (örn: Star Wars (1977))")
    if st.button("Öner"):
        if title:
            if option.endswith("Başlık)"):
                sim = tfidf_title_model()
            elif option.endswith("Tür)"):
                sim = tfidf_genre_model()
            else:
                sim = tfidf_hybrid_model()
            results = content_recommend(title, sim)
            for i, movie in enumerate(results, 1):
                st.write(f"{i}. {movie}")
        else:
            st.warning("Lütfen bir film adı girin.")

elif option.startswith("Kişiselleştirilmiş"):
    user_id = st.number_input("Kullanıcı ID girin (1-943)", min_value=1, max_value=943, step=1)
    if st.button("Öner"):
        fav_genres = get_user_genre_profile(user_id)
        st.write(f"🎯 Favori türler: {', '.join(fav_genres)}")
        results = recommend_by_genre(fav_genres)
        for i, movie in enumerate(results, 1):
            st.write(f"{i}. {movie}")

elif option.startswith("Zaman"):
    start = st.number_input("Başlangıç yılı", min_value=1900, max_value=2025, value=1990, step=1)
    end = st.number_input("Bitiş yılı", min_value=1900, max_value=2025, value=2000, step=1)
    if st.button("Öner"):
        results = recommend_by_year(start, end)
        for i, movie in enumerate(results, 1):
            st.write(f"{i}. {movie}")

elif option.startswith("Kullanıcı"):
    user_id = st.number_input("Kullanıcı ID girin (1-943)", min_value=1, max_value=943, step=1)
    if st.button("Öner"):
        results = collab_recommend(user_id)
        if len(results) == 0:
            st.warning("Öneri bulunamadı.")
        else:
            for i, movie in enumerate(results, 1):
                st.write(f"{i}. {movie}")
