"""Database configuration for Cerebro X.

DATABASE_URL is read from the environment variable DATABASE_URL.
Defaults to SQLite for local development.

For Docker/production, set DATABASE_URL in docker-compose.yml or environment:
  DATABASE_URL=postgresql://user:pass@db:5432/cerebro_x

SQLite connection requires check_same_thread=False.
PostgreSQL does not need connect_args.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cerebro_x.db")

# SQLite needs check_same_thread=False for multi-threaded FastAPI use.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — provides a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
