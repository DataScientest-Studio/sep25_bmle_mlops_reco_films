import sqlite3
import sys

DB_PATH = "data/db/staging.sqlite"


def run_check(conn, name, query):
    cur = conn.execute(query)
    value = cur.fetchone()[0]

    if value == 0:
        print(f"✅ {name}")
        return True
    else:
        print(f"❌ {name} → {value} problème(s) détecté(s)")
        return False


def main():
    print("🔍 Démarrage des checks qualité sur la DB\n")

    conn = sqlite3.connect(DB_PATH)
    checks = []

    # 1. Ratings hors [0,5]
    checks.append(run_check(
        conn,
        "Ratings dans [0,5]",
        """
        SELECT COUNT(*)
        FROM raw_ratings
        WHERE rating < 0 OR rating > 5 OR rating IS NULL
        """
    ))

    # 2. userId NULL
    checks.append(run_check(
        conn,
        "userId non NULL",
        """
        SELECT COUNT(*)
        FROM raw_ratings
        WHERE userId IS NULL
        """
    ))

    # 3. movieId NULL
    checks.append(run_check(
        conn,
        "movieId non NULL",
        """
        SELECT COUNT(*)
        FROM raw_ratings
        WHERE movieId IS NULL
        """
    ))

    # 4. Intégrité référentielle movieId
    checks.append(run_check(
        conn,
        "ratings.movieId existe dans movies",
        """
        SELECT COUNT(*)
        FROM raw_ratings r
        LEFT JOIN raw_movies m ON r.movieId = m.movieId
        WHERE m.movieId IS NULL
        """
    ))

    # 5. Doublons exacts userId/movieId/timestamp
    checks.append(run_check(
        conn,
        "Pas de doublons (userId, movieId, timestamp)",
        """
        SELECT COUNT(*)
        FROM (
            SELECT userId, movieId, timestamp, COUNT(*) AS c
            FROM raw_ratings
            GROUP BY userId, movieId, timestamp
            HAVING c > 1
        )
        """
    ))

    conn.close()

    print("\n📊 Résumé :")
    if all(checks):
        print("🎉 Tous les checks sont OK")
        sys.exit(0)
    else:
        print("⛔ Échec des checks qualité — pipeline stoppé")
        sys.exit(1)


if __name__ == "__main__":
    main()
