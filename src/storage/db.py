import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import UUID

import sqlite_utils

from src.config import settings
from src.models.diffusion import CalibrationResult, DiffusionFit
from src.models.market import KalshiMarket, PriceSnapshot
from src.models.news import NewsArticle
from src.models.scoring import JevScore

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or settings.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = None
        self._init_schema()

    @property
    def db(self) -> sqlite_utils.Database:
        if self._db is None:
            self._db = sqlite_utils.Database(self.db_path)
        return self._db

    def _init_schema(self):
        """Create tables if they don't exist"""
        # Markets table
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                ticker TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT,
                status TEXT,
                yes_bid INTEGER,
                no_bid INTEGER,
                last_price INTEGER,
                volume INTEGER,
                open_time TEXT,
                close_time TEXT,
                settlement_time TEXT,
                result INTEGER,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Price snapshots
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS price_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                yes_bid INTEGER,
                no_bid INTEGER,
                last_price INTEGER,
                volume INTEGER,
                FOREIGN KEY (ticker) REFERENCES markets (ticker)
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_price_ticker_time ON price_snapshots (ticker, timestamp)")

        # News articles
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id TEXT PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                summary TEXT,
                published_at TEXT,
                fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                author TEXT,
                metadata TEXT,
                scored INTEGER DEFAULT 0,
                score_id TEXT
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles (published_at)")

        # Jev scores
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS jev_scores (
                id TEXT PRIMARY KEY,
                article_id TEXT NOT NULL,
                market_ticker TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                relevance REAL NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT,
                model_version TEXT,
                prompt_version TEXT,
                latency_ms INTEGER,
                FOREIGN KEY (article_id) REFERENCES news_articles (id),
                FOREIGN KEY (market_ticker) REFERENCES markets (ticker)
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_scores_article ON jev_scores (article_id)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_scores_market ON jev_scores (market_ticker)")

        # Diffusion fits
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS diffusion_fits (
                id TEXT PRIMARY KEY,
                score_id TEXT NOT NULL,
                article_id TEXT NOT NULL,
                market_ticker TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                news_time TEXT NOT NULL,
                price_at_news REAL NOT NULL,
                amplitude REAL,
                timescale REAL,
                baseline REAL,
                half_life REAL,
                r_squared REAL,
                n_points INTEGER,
                rmse REAL,
                fit_start TEXT,
                fit_end TEXT,
                FOREIGN KEY (score_id) REFERENCES jev_scores (id),
                FOREIGN KEY (article_id) REFERENCES news_articles (id),
                FOREIGN KEY (market_ticker) REFERENCES markets (ticker)
            )
        """)

        # Calibration results
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS calibration_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_ticker TEXT NOT NULL,
                model_version TEXT,
                evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                n_samples INTEGER,
                brier_score REAL,
                brier_decomposition TEXT,
                reliability_bins TEXT,
                ece REAL,
                mce REAL
            )
        """)

        logger.info("Database schema initialized")

    # --- Market operations ---
    def upsert_market(self, market: KalshiMarket):
        self.db["markets"].upsert(market.model_dump(mode="json"), pk="ticker")

    def get_market(self, ticker: str) -> KalshiMarket | None:
        row = self.db["markets"].get(ticker)
        if row:
            return KalshiMarket(**row)
        return None

    # --- Price snapshot operations ---
    def insert_price_snapshot(self, snapshot: PriceSnapshot):
        self.db["price_snapshots"].insert(snapshot.model_dump(mode="json"))

    def get_price_snapshots(self, ticker: str, since: datetime | None = None, limit: int | None = None) -> list[PriceSnapshot]:
        query = "SELECT * FROM price_snapshots WHERE ticker = ?"
        params = [ticker]
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        query += " ORDER BY timestamp ASC"
        if limit:
            query += f" LIMIT {limit}"
        rows = self.db.query(query, params)
        return [PriceSnapshot(**r) for r in rows]

    # --- News article operations ---
    def insert_article(self, article: NewsArticle) -> bool:
        """Insert article, return True if new, False if duplicate URL"""
        try:
            self.db["news_articles"].insert(article.model_dump(mode="json"), pk="id", replace=False)
        except sqlite_utils.db.NotUniqueError:
            return False
        else:
            return True

    def get_unscored_articles(self, limit: int = 100) -> list[NewsArticle]:
        rows = self.db.query(
            "SELECT * FROM news_articles WHERE scored = 0 ORDER BY published_at DESC LIMIT ?",
            [limit],
        )
        return [NewsArticle(**r) for r in rows]

    def mark_article_scored(self, article_id: UUID, score_id: UUID):
        self.db["news_articles"].update(str(article_id), {"scored": 1, "score_id": str(score_id)})

    def get_article(self, article_id: UUID) -> NewsArticle | None:
        row = self.db["news_articles"].get(str(article_id))
        if row:
            return NewsArticle(**row)
        return None

    # --- Jev score operations ---
    def insert_score(self, score: JevScore):
        self.db["jev_scores"].insert(score.model_dump(mode="json"), pk="id")

    def get_scores_for_market(self, ticker: str, limit: int = 1000) -> list[JevScore]:
        rows = self.db.query(
            "SELECT * FROM jev_scores WHERE market_ticker = ? ORDER BY created_at DESC LIMIT ?",
            [ticker, limit],
        )
        return [JevScore(**r) for r in rows]

    def get_score(self, score_id: UUID) -> JevScore | None:
        row = self.db["jev_scores"].get(str(score_id))
        if row:
            return JevScore(**row)
        return None

    # --- Diffusion fit operations ---
    def insert_diffusion_fit(self, fit: DiffusionFit):
        self.db["diffusion_fits"].insert(fit.model_dump(mode="json"), pk="id")

    def get_diffusion_fits(self, market_ticker: str, limit: int = 1000) -> list[DiffusionFit]:
        rows = self.db.query(
            "SELECT * FROM diffusion_fits WHERE market_ticker = ? ORDER BY created_at DESC LIMIT ?",
            [market_ticker, limit],
        )
        return [DiffusionFit(**r) for r in rows]

    # --- Calibration operations ---
    def insert_calibration(self, result: CalibrationResult) -> int:
        data = result.model_dump(mode="json")
        data["brier_decomposition"] = json.dumps(data["brier_decomposition"])
        data["reliability_bins"] = json.dumps(data["reliability_bins"])
        return self.db["calibration_results"].insert(data)

    def get_latest_calibration(self, market_ticker: str) -> CalibrationResult | None:
        row = self.db.query(
            "SELECT * FROM calibration_results WHERE market_ticker = ? ORDER BY evaluated_at DESC LIMIT 1",
            [market_ticker],
        )
        if row:
            r = row[0]
            r["brier_decomposition"] = json.loads(r["brier_decomposition"])
            r["reliability_bins"] = json.loads(r["reliability_bins"])
            return CalibrationResult(**r)
        return None


# Global instance
db = Database()

