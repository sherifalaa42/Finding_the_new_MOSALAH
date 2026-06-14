# ⚽ Finding the Next Mo Salah

A data-driven football scouting system that identifies emerging talent with a statistical profile similar to Mohamed Salah. Using a dataset of 5,600+ players, this project applies machine learning and similarity analysis to surface high-potential wingers from outside the top clubs.

## Overview

The goal was to find a young, underrated forward who mirrors Salah's key attributes — pace, dribbling, finishing, vision — but hasn't yet broken into a top-tier club. The pipeline filters, scores, and ranks players using a combination of weighted feature engineering and unsupervised ML.

## How it works

- **Data cleaning** — removed nulls, fixed data types, dropped duplicates, and filtered out goalkeepers and players outside the 19–27 age range
- **Position scoring** — engineered weighted attacking, midfield, and defensive scores per player to classify positions without a position column in the raw data
- **Star player index** — combined position scores (60% attacking, 25% midfield, 15% defensive) into a single composite score and ranked all players by percentile
- **Similarity to Salah** — used Euclidean distance on StandardScaler-normalised features to measure how close each player is to Salah's statistical profile
- **Clustering & visualisation** — applied K-Means clustering and t-SNE dimensionality reduction to group and plot similar players, then compared finalists with radar charts, heatmaps, and bar plots

## Tech stack

| Area | Libraries |
|---|---|
| Data | pandas, NumPy |
| ML | scikit-learn (KMeans, t-SNE, PCA) |
| Visualisation | Matplotlib, Seaborn |
| Preprocessing | StandardScaler, OrdinalEncoder |

## Results

After filtering for attackers in the top 5th percentile not playing for elite clubs, the model identified the following players as closest to Salah's profile:

- Oyarzabal
- Martin Terrier
- Amine Gouiri
- **Rafael Leão** — highlighted as the strongest candidate given his age and room for growth

## Getting started

```bash
git clone [https://github.com/your-username/finding-next-salah](https://github.com/sherifalaa42/Finding_the_new_MOSALAH)
cd Finding_the_new_MOSALAH
pip install -r requirements.txt
jupyter notebook Analysis.ipynb
```

## Dataset

`player_stats.csv` — 5,682 players with 41 attributes including pace, shooting, dribbling, passing, physicality, club, country, age, and market value.

## Custom player input

The notebook includes a template at the bottom to input any player's stats and run them through the full pipeline to see how they compare to Salah.
