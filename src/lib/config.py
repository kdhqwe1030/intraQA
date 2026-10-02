import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.4-mini")
LAW_OC = os.getenv("LAW_OC", "")


def pg_url(driver: str = "psycopg") -> str:
    """SQLAlchemy 형식 접속 URL. langchain-postgres는 postgresql+psycopg:// 를 쓴다."""
    user = os.getenv("POSTGRES_USER", "intraqa")
    password = os.getenv("POSTGRES_PASSWORD", "intraqa")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5445")
    db = os.getenv("POSTGRES_DB", "intraqa")
    prefix = f"postgresql+{driver}" if driver else "postgresql"
    return f"{prefix}://{user}:{password}@{host}:{port}/{db}"
