from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app import config

config.ensure_dirs()

engine = create_engine(
    f"sqlite:///{config.DB_PATH}",
    connect_args={"check_same_thread": False, "timeout": 30},
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")  # required for ON DELETE CASCADE
    cur.execute("PRAGMA journal_mode=WAL")  # API reads while the worker writes
    cur.close()


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from app import models  # noqa: F401  (registers tables)

    Base.metadata.create_all(engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    with SessionLocal() as session:
        yield session


@contextmanager
def session_scope() -> Iterator[Session]:
    """For background code: commits on success, rolls back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
