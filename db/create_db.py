from sqlalchemy import create_engine
from models import Base
from config import ConfigManager, DBConfig
from db import db

def main():
    print("Creation des tables...")
    Base.metadata.create_all(db)
    print("Tout est terminé !")

if __name__ == '__main__':
    main()