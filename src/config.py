from sqlalchemy import create_engine

DATABASE = "sqlite:///netflix_clone.db"

engine = create_engine(DATABASE, echo=False)