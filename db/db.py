from sqlalchemy import create_engine
from config import config_manager

db = create_engine(config_manager.config.connection)