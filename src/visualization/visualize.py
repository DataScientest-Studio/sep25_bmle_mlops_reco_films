# ------------------------------------------------------------
# FICHIER : src/visualization/visualize.py
# ------------------------------------------------------------
# OBJECTIF :
# - Analyse exploratoire MovieLens (EDA)
# - Plaquette 1 : 3 tableaux annuels + TOTAL
# - Plaquette 2 : 6 graphes descriptifs
#
# DÉFINITION :
# - 1 session = (userId, jour)
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
# 0) Chargement des données
# ------------------------------------------------------------
DATA_DIR = Path("data/processed")

ratings = pd.read_csv(DATA_DIR / "ratings_clean.csv")
movies = pd.read_csv(DATA_DIR / "movies_clean.csv")

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
# 1) BASE SESSIONS (user, jour)
# ------------------------------------------------------------
sessions = (
    ratings
    .groupby(["year", "month", "date", "userId"])
    .agg(nb_notes=("rating", "count"))
    .reset_index()
)

# ------------------------------------------------------------
# 2) TABLEAUX ANNUELS
# ------------------------------------------------------------
def compute_yearly_stats_sessions(ratings, movies, sessions):

    first_user_year = ratings.groupby("userId")["year"].min()
    first_movie_year = ratings.groupby("movieId")["year"].min()

    genre_count = (
        movies["genres"]
        .fillna("")
        .replace({"(no genres listed)": ""})
        .str.split("|")
        .apply(len)
    )
    genre_count.index = movies["movieId"]

    rows = []

    for year, g in ratings.groupby("year"):
        sess_y = sessions[sessions["year"] == year]

        note_dist = (
            g["rating"]
            .value_counts(normalize=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0)
            * 100
        )

        rows.append({
            "year": year,
            "nb_notes": len(g),
            "nb_noteurs": g["userId"].nunique(),
            "nb_films_notes": g["movieId"].nunique(),
            "nb_nouveaux_noteurs": (first_user_year == year).sum(),
            "nb_nouveaux_films": (first_movie_year == year).sum(),
            "nb_moyen_genres_par_film": genre_count.loc[g["movieId"].unique()].mean(),
            "nb_moyen_notes_par_noteur": len(g) / g["userId"].nunique(),
            "nb_moyen_notes_par_session": sess_y["nb_notes"].mean(),
            "note_moyenne": g["rating"].mean(),
            "pct_notes_1": note_dist.loc[1],
            "pct_notes_2": note_dist.loc[2],
            "pct_notes_3": note_dist.loc[3],
            "pct_notes_4": note_dist.loc[4],
            "pct_notes_5": note_dist.loc[5],
        })

    df_year = pd.DataFrame(rows).sort_values("year")

    total_note_dist = (
        ratings["rating"]
        .value_counts(normalize=True)
        .reindex([1, 2, 3, 4, 5], fill_value=0)
        * 100
    )

    total = {
        "year": "TOTAL",
        "nb_notes": len(ratings),
        "nb_noteurs": ratings["userId"].nunique(),
        "nb_films_notes": ratings["movieId"].nunique(),
        "nb_nouveaux_noteurs": first_user_year.nunique(),
        "nb_nouveaux_films": first_movie_year.nunique(),
        "nb_moyen_genres_par_film": genre_count.loc[ratings["movieId"].unique()].mean(),
        "nb_moyen_notes_par_noteur": len(ratings) / ratings["userId"].nunique(),
        "nb_moyen_notes_par_session": sessions["nb_notes"].mean(),
        "note_moyenne": ratings["rating"].mean(),
        "pct_notes_1": total_note_dist.loc[1],
        "pct_notes_2": total_note_dist.loc[2],
        "pct_notes_3": total_note_dist.loc[3],
        "pct_notes_4": total_note_dist.loc[4],
        "pct_notes_5": total_note_dist.loc[5],
    }

    return pd.concat([df_year, pd.DataFrame([total])], ignore_index=True)

COLUMN_LABELS = {
    "year": "Année",

    "nb_notes": "Notes",
    "nb_noteurs": "Noteurs",
    "nb_films_notes": "Films notés",
    "nb_nouveaux_noteurs": "Nvx noteurs",
    "nb_nouveaux_films": "Nvx films",

    "nb_moyen_genres_par_film": "genres / film",
    "nb_moyen_notes_par_noteur": "notes / noteur",
    "nb_moyen_notes_par_session": "notes / session",

    "note_moyenne": "Note moyenne",
    "pct_notes_1": "% notes 1",
    "pct_notes_2": "% notes 2",
    "pct_notes_3": "% notes 3",
    "pct_notes_4": "% notes 4",
    "pct_notes_5": "% notes 5",
}


def display_yearly_stats_figure(df, show=False):
    """
    Plaquette 1 : 3 tableaux avec titre AU-DESSUS
    """

    df = df.copy()
    df["year"] = df["year"].astype(str)

    tables = {
        "Volumes": [
            "nb_notes", "nb_noteurs", "nb_films_notes",
            "nb_nouveaux_noteurs", "nb_nouveaux_films"
        ],
        "Moyennes": [
            "nb_moyen_genres_par_film",
            "nb_moyen_notes_par_noteur",
            "nb_moyen_notes_par_session",
        ],
        "Notes & répartition": [
            "note_moyenne",
            "pct_notes_1", "pct_notes_2",
            "pct_notes_3", "pct_notes_4", "pct_notes_5",
        ],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # TITRE AU-DESSUS DES TABLEAUX
    fig.text(
        0.5, 0.98,
        "MovieLens – Statistiques annuelles",
        ha="center", va="top",
        fontsize=14, fontweight="bold"
    )

    for ax, (title, cols) in zip(axes, tables.items()):
        ax.axis("off")
         # position de l'axe dans la figure
        bbox = ax.get_position()

        # titre AU-DESSUS du tableau
        fig.text(
            bbox.x0 + bbox.width / 2,
            bbox.y1 + 0.02,      
            title,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        sub = df[["year"] + cols].round(2)

        display_labels = [
        COLUMN_LABELS.get(c, c) for c in sub.columns
        ]

        # largeur relative des colonnes
        # 1ère colonne (année) plus étroite
        # autres colonnes plus larges
        n_cols = len(sub.columns)

        col_widths = [0.12] + [0.22] * (n_cols - 1)

        table = ax.table(
            cellText=sub.values,
            colLabels=display_labels,
            colWidths=col_widths,
            cellLoc="center",
            loc="center",
        )

        table.scale(1, 1.4)
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if show:
        plt.show()

    return fig


# ------------------------------------------------------------
# 3) CALCUL + AFFICHAGE PLAQUETTE 1
# ------------------------------------------------------------
stats_year = compute_yearly_stats_sessions(ratings, movies, sessions)
fig_tables = display_yearly_stats_figure(stats_year, show=False)

# ------------------------------------------------------------
# 4) PARAMÈTRES GRAPHIQUES
# ------------------------------------------------------------
PARTIAL_YEAR = ratings["year"].max()
YEARS_COMPARE = [
    PARTIAL_YEAR - 3,
    PARTIAL_YEAR - 2,
    PARTIAL_YEAR - 1,
    PARTIAL_YEAR,
]

# ------------------------------------------------------------
# 5) DONNÉES POUR GRAPHES
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

daily_voters = (
    ratings.groupby(["year", "month", "date"])["userId"]
    .nunique()
    .reset_index(name="nb_voters")
)

avg_daily_voters = (
    daily_voters.groupby(["year", "month"])["nb_voters"]
    .mean()
    .reset_index(name="avg_daily_voters")
)

avg_notes_per_session = (
    sessions.groupby(["year", "month"])["nb_notes"]
    .mean()
    .reset_index(name="avg_notes_per_session")
)

tmp = ratings.sort_values(["userId", "datetime"])
tmp["prev_dt"] = tmp.groupby("userId")["datetime"].shift(1)
tmp["is_new_12m"] = tmp["prev_dt"].isna() | ((tmp["datetime"] - tmp["prev_dt"]).dt.days > 365)

new_voters_month = (
    tmp[tmp["year"].isin(YEARS_COMPARE)]
    .groupby(["year", "month"])["userId"]
    .nunique()
    .reset_index(name="nb_new_voters")
)

genres_oh = (
    movies["genres"]
    .fillna("")
    .replace({"(no genres listed)": ""})
    .str.get_dummies(sep="|")
)

movies_genres = pd.concat([movies[["movieId"]], genres_oh], axis=1)
ratings_genres = ratings.merge(movies_genres, on="movieId", how="inner")

genre_volume_ref = (
    ratings_genres[ratings_genres["year"] == PARTIAL_YEAR][genres_oh.columns]
    .sum()
    .sort_values(ascending=False)
    .head(12)
)

TOP_12_GENRES = genre_volume_ref.index.tolist()
TOP_9_GENRES = TOP_12_GENRES[:9]

genre_counts = (
    ratings_genres[ratings_genres["year"].isin(YEARS_COMPARE)]
    .groupby("year")[TOP_12_GENRES]
    .sum()
)

genre_pct = genre_counts.div(genre_counts.sum(axis=1), axis=0) * 100

# ------------------------------------------------------------
# 6) PLAQUETTE 2 – 6 GRAPHES (Graphe 6 conservé)
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(22, 10))

# (1)
ax = axes[0, 0]
ax.plot(votes_full_year["year"], votes_full_year["nb_votes_k"], marker="o")
ax.set_title("Nombre de votes par année pleine (en milliers)")
ax.set_xticks(votes_full_year["year"])
ax.set_xticklabels(votes_full_year["year"].astype(str))
ax.tick_params(axis="x", rotation=45)

# (2)
ax = axes[0, 1]
for y in YEARS_COMPARE:
    d = avg_daily_voters[avg_daily_voters["year"] == y]
    ax.plot(d["month"], d["avg_daily_voters"], marker="o", label=str(y))
ax.set_title("Moyenne journalière du nombre de votants")
ax.set_xticks(range(1, 13))
ax.set_xticklabels([MONTH_LABELS[m] for m in range(1, 13)])
ax.legend()

# (3)
ax = axes[0, 2]
for y in YEARS_COMPARE:
    d = avg_notes_per_session[avg_notes_per_session["year"] == y]
    ax.plot(d["month"], d["avg_notes_per_session"], marker="o", label=str(y))
ax.set_title("Nombre moyen de notes par session")
ax.set_xticks(range(1, 13))
ax.set_xticklabels([MONTH_LABELS[m] for m in range(1, 13)])
ax.legend()

# (4)
ax = axes[1, 0]
for y in YEARS_COMPARE:
    d = new_voters_month[new_voters_month["year"] == y]
    ax.plot(d["month"], d["nb_new_voters"], marker="o", label=str(y))
ax.set_title("Nouveaux votants par mois")
ax.set_xticks(range(1, 13))
ax.set_xticklabels([MONTH_LABELS[m] for m in range(1, 13)])
ax.legend()

# (5)
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

# (6) 
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

ax.set_xlim(-0.5, len(x) - 0.5 + 0.75)

handles, labels_leg = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1],
    labels_leg[::-1],
    title="Genres",
    fontsize=7,
    title_fontsize=8,
    loc="center left",
    bbox_to_anchor=(0.75, 0.5),
    borderaxespad=0.0,
)


plt.suptitle("MovieLens – Analyse exploratoire", fontsize=14)
plt.tight_layout()

# ------------------------------------------------------------
# 7) AFFICHAGE SIMULTANÉ DES DEUX PLAQUETTES
# ------------------------------------------------------------
plt.show()

print("\n✅ Analyse exploratoire générée avec succès")
