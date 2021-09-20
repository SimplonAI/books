from sqlalchemy import create_engine
from models import Base
from config import ConfigManager, DBConfig


def main():
    print("Chargement...")
    config_manager = ConfigManager()
    print(config_manager.config)
    print("Connexion à la bdd...")
    engine = create_engine(config_manager.config.connection)
    print("Creation des tables...")
    Base.metadata.create_all(engine)
    print("Tout est terminé !")

if __name__ == '__main__':
    main()