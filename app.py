import streamlit as st
import pandas as pd
from model import PoissonPredictor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("⚽ Prediksi Skor Akhir - EPL")

# Upload file CSV baru
st.sidebar.title("📤 Input Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV Baru", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset berhasil dimuat.")
else:
    data = pd.read_csv("epl_matches.csv")

model = PoissonPredictor(data)

teams = sorted(data['HomeTeam'].unique())
home_team = st.selectbox("🏠 Tim Kandang", teams)
away_team = st.selectbox("✈️ Tim Tandang", teams)

if st.button("Prediksi Skor"):
    if home_team == away_team:
        st.warning("Tim kandang dan tandang tidak boleh sama.")
    else:
        home_score, away_score = model.predict_score(home_team, away_team)
        st.success(f"Prediksi Skor: **{home_team} {home_score} - {away_score} {away_team}**")

        st.subheader("📊 Probabilitas Skor")
        score_matrix = model.predict_score_distribution(home_team, away_team)

        fig, ax = plt.subplots()
        sns.heatmap(score_matrix, annot=True, fmt=".1%", cmap="Blues", 
                    xticklabels=range(6), yticklabels=range(6), ax=ax)
        ax.set_xlabel(f"{away_team} Goals")
        ax.set_ylabel(f"{home_team} Goals")
        st.pyplot(fig)

        home_win = np.sum(np.tril(score_matrix, -1))
        draw = np.sum(np.diag(score_matrix))
        away_win = np.sum(np.triu(score_matrix, 1))

        st.subheader("🧠 Probabilitas Hasil Akhir")
        st.write(f"- **{home_team} Menang:** {home_win:.1%}")
        st.write(f"- **Seri:** {draw:.1%}")
        st.write(f"- **{away_team} Menang:** {away_win:.1%}")