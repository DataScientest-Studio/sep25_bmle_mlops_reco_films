# ------------------------------------------------------------
# FICHIER : src/visualization/visualize.py
# ------------------------------------------------------------
# OBJECTIF :
# - Analyse exploratoire MovieLens (EDA)
# - Plaque de 6 graphes descriptifs
#
# DONNÉES :
# - data/processed/ratings_clean.csv
# - data/processed/movies_clean.csv
# ------------------------------------------------------------

from pathlib import Path
import calendar
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Chargement des données
# ------------------------------------------------------------
DATA_DIR = Path("data/processed")

ratings = pd.read_csv(DATA_DIR / "ratings_clean.csv")
movies = pd.read_csv(DATA_DIR / "movies_clean.csv")

# ------------------------------------------------------------
# 2) Préparation temporelle
# ------------------------------------------------------------
ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
ratings["date"] = ratings["datetime"].dt.date
ratings["year"] = ratings["datetime"].dt.year.astype(int)
ratings["month"] = ratings["datetime"].dt.month.astype(int)
ratings["week"] = ratings["datetime"].dt.isocalendar().week.astype(int)

MONTH_LABELS = {
    1: "JAN", 2: "FEV", 3: "MAR", 4: "AVR",
    5: "MAI", 6: "JUN", 7: "JUI", 8: "AOU",
    9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
}

# ------------------------------------------------------------
# 3) Paramètres EDA
# ------------------------------------------------------------
PARTIAL_YEAR = ratings["year"].max()

YEARS_COMPARE = [
    PARTIAL_YEAR - 3,
    PARTIAL_YEAR - 2,
    PARTIAL_YEAR - 1,
    PARTIAL_YEAR,
]

YEAR_COLORS = {
    YEARS_COMPARE[0]: "#d62728",
    YEARS_COMPARE[1]: "#2ca02c",
    YEARS_COMPARE[2]: "#ff7f0e",
    YEARS_COMPARE[3]: "#1f77b4",
}

# ------------------------------------------------------------
# 4) Graphe 1 – Votes par année pleine
# ------------------------------------------------------------
year_bounds = ratings.groupby("year")["datetime"].agg(["min", "max"])

full_years = year_bounds[
    (year_bounds["min"].dt.month == 1) &
    (year_bounds["min"].dt.day == 1) &
    (year_bounds["max"].dt.month == 12) &
    (year_bounds["max"].dt.day == 31)
].index.astype(int)

votes_full_year = (
    ratings[ratings["year"].isin(full_years)]
    .groupby("year")
    .size()
    .reset_index(name="nb_votes")
)

votes_full_year["nb_votes_k"] = votes_full_year["nb_votes"] / 1_000

# ------------------------------------------------------------
# 5) Graphe 2 – Moyenne journalière du nombre de votants
# ------------------------------------------------------------
daily_voters = (
    ratings
    .groupby(["year", "month", "date"])["userId"]
    .nunique()
    .reset_index(name="nb_voters")
)

avg_daily_voters = (
    daily_voters
    .groupby(["year", "month"])["nb_voters"]
    .mean()
    .reset_index(name="avg_daily_voters")
)

# ------------------------------------------------------------
# 6) Graphe 3 – Moyenne journalière des notes par votant
# ------------------------------------------------------------
daily_stats = (
    ratings
    .groupby(["year", "month", "date"])
    .agg(
        nb_votes=("rating", "count"),
        nb_voters=("userId", "nunique"),
    )
    .reset_index()
)

daily_stats["notes_per_voter"] = (
    daily_stats["nb_votes"]
    / daily_stats["nb_voters"].clip(lower=1)
)

avg_daily_notes_per_voter = (
    daily_stats
    .groupby(["year", "month"])["notes_per_voter"]
    .mean()
    .reset_index(name="avg_daily_notes_per_voter")
)

# ------------------------------------------------------------
# 7) Graphe 4 – Nouveaux votants
# ------------------------------------------------------------
tmp = ratings.sort_values(["userId", "datetime"])
tmp["prev_dt"] = tmp.groupby("userId")["datetime"].shift(1)
tmp["delta_days"] = (tmp["datetime"] - tmp["prev_dt"]).dt.days
tmp["is_new_12m"] = tmp["prev_dt"].isna() | (tmp["delta_days"] > 365)

new_voters_month = (
    tmp[tmp["is_new_12m"] & tmp["year"].isin(YEARS_COMPARE)]
    .groupby(["year", "month"])["userId"]
    .nunique()
    .reset_index(name="nb_new_voters")
)

# ------------------------------------------------------------
# 8) Genres – TOP 12
# ------------------------------------------------------------
genres_oh = (
    movies["genres"]
    .fillna("")
    .replace({"(no genres listed)": ""})
    .str.get_dummies(sep="|")
)

movies_genres = pd.concat([movies[["movieId"]], genres_oh], axis=1)
ratings_genres = ratings.merge(movies_genres, on="movieId", how="inner")

genre_volume_ref = (
    ratings_genres[ratings_genres["year"] == PARTIAL_YEAR]
    [genres_oh.columns]
    .sum()
    .sort_values(ascending=False)
    .head(12)
)

TOP_12_GENRES = genre_volume_ref.index.tolist()
TOP_9_GENRES = TOP_12_GENRES[:9]

# ------------------------------------------------------------
# 9) Graphe 5 – % de votes par genre
# ------------------------------------------------------------
genre_counts = (
    ratings_genres[ratings_genres["year"].isin(YEARS_COMPARE)]
    .groupby("year")[TOP_12_GENRES]
    .sum()
)

genre_pct = genre_counts.div(genre_counts.sum(axis=1), axis=0) * 100

# ------------------------------------------------------------
# 10) Graphe 6 – Nouveaux films M et M-1 (par genre)
# ------------------------------------------------------------
first_seen = (
    ratings.groupby("movieId")["datetime"]
    .min()
    .reset_index(name="first_seen")
)

first_seen["year"] = first_seen["first_seen"].dt.year
first_seen["month"] = first_seen["first_seen"].dt.month

last_year = first_seen["year"].max()
last_month = first_seen.loc[first_seen["year"] == last_year, "month"].max()

if last_month == 1:
    prev_year = last_year - 1
    prev_month = 12
else:
    prev_year = last_year
    prev_month = last_month - 1

months_focus = [(prev_year, prev_month), (last_year, last_month)]

new_movies_2m = (
    first_seen[first_seen[["year", "month"]].apply(tuple, axis=1).isin(months_focus)]
    .merge(movies_genres, on="movieId", how="inner")
)

new_movies_genre_2m = (
    new_movies_2m
    .groupby(["year", "month"])[TOP_9_GENRES]
    .sum()
)

new_movies_genre_2m["Autres"] = (
    new_movies_2m
    .groupby(["year", "month"])[genres_oh.columns]
    .sum()
    .sum(axis=1)
    - new_movies_genre_2m.sum(axis=1)
)

# ------------------------------------------------------------
# 11) FIGURE – 6 GRAPHES
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(22, 10))

# (1) Votes par année pleine
ax = axes[0, 0]

years = votes_full_year["year"].astype(int).tolist()
values = votes_full_year["nb_votes_k"].tolist()

ax.plot(years, values, marker="o")

ax.set_title("Nombre de votes par année pleine (en milliers)")
ax.set_ylabel("Votes (k)")

# 🔑 FORCER un axe discret
ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years])

ax.tick_params(axis="x", rotation=45, labelsize=7)
ax.tick_params(axis="y", labelsize=7)


# (2) Moyenne journalière du nombre de votants
ax = axes[0, 1]
for y in YEARS_COMPARE:
    d = avg_daily_voters[avg_daily_voters["year"] == y]
    ax.plot(d["month"], d["avg_daily_voters"], marker="o", label=str(y))
ax.set_title("Moyenne journalière du nombre de votants")
ax.set_xticks(range(1, 13))
ax.set_xticklabels([MONTH_LABELS[m] for m in range(1, 13)])
ax.legend()

# (3) Moyenne journalière des notes par votant
ax = axes[0, 2]
for y in YEARS_COMPARE:
    d = avg_daily_notes_per_voter[avg_daily_notes_per_voter["year"] == y]
    ax.plot(d["month"], d["avg_daily_notes_per_voter"], marker="o", label=str(y))
ax.set_title("Moyenne journalière des notes par votant")
ax.set_xticks(range(1, 13))
ax.set_xticklabels([MONTH_LABELS[m] for m in range(1, 13)])
ax.legend()

# (4) Nouveaux votants
ax = axes[1, 0]
for y in YEARS_COMPARE:
    d = new_voters_month[new_voters_month["year"] == y]
    ax.plot(d["month"], d["nb_new_voters"], marker="o", label=str(y))
ax.set_title("Nouveaux votants par mois")
ax.set_xticks(range(1, 13))
ax.set_xticklabels([MONTH_LABELS[m] for m in range(1, 13)])
ax.legend()

# (5) Top genres – % votes
ax = axes[1, 1]
bar_h = 0.18
y_pos = range(len(TOP_12_GENRES))
for i, y in enumerate(YEARS_COMPARE):
    ax.barh(
        [p + i * bar_h for p in y_pos],
        genre_pct.loc[y],
        height=bar_h,
        label=str(y),
    )
ax.set_yticks([p + 1.5 * bar_h for p in y_pos])
ax.set_yticklabels(TOP_12_GENRES)
ax.set_title("Top genres – % votes")
ax.legend()

# (6) Nouveaux films – mois courant vs précédent (par genre)
ax = axes[1, 2]

labels = [f"{MONTH_LABELS[m]} {y}" for y, m in months_focus]
x = list(range(len(labels)))
bottom = [0] * len(labels)
bar_width = 0.6

stack_order = TOP_9_GENRES + ["Autres"]

for g in stack_order:
    vals = [
        new_movies_genre_2m.loc[(y, m), g]
        if (y, m) in new_movies_genre_2m.index
        else 0
        for (y, m) in months_focus
    ]

    ax.bar(
        x,
        vals,
        bottom=bottom,
        width=bar_width,
        label=g,
    )

    # étiquettes par GENRE dans les segments
    for i, (b, v) in enumerate(zip(bottom, vals)):
        if v > 0:
            ax.text(
                i,
                b + v / 2,
                f"{int(v)}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )

    bottom = [b + v for b, v in zip(bottom, vals)]

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Nombre de nouveaux films")
ax.set_title("Nouveaux films – mois courant vs précédent (par genre)")

# --- réserver 25 % de la largeur de l’axe pour la légende
ax.set_xlim(-0.5, len(x) - 0.5 + 0.75)

# --- légende DANS l’axe, sans superposition
handles, labels_leg = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels_leg[::-1],
    title="Genres",
    fontsize=7,
    title_fontsize=8,
    loc="center left",
    bbox_to_anchor=(0.75, 0.5),   # début du quart droit
    borderaxespad=0.0,
)

plt.suptitle("MovieLens – Analyse exploratoire", fontsize=14)
plt.tight_layout()
plt.show()

print("\n✅ Analyse exploratoire générée avec succès")
