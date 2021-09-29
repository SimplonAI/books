from sqlalchemy import create_engine
from db.models import Base
from db.config import ConfigManager, DBConfig


def create_db(db):
    print("Creation des tables...")
    Base.metadata.create_all(db)
    print("Tout est terminé !")


if __name__ == "__main__":
    from db import db
    create_db(db)
