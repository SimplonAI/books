from sqlalchemy import create_engine, Column, Integer, ForeignKey, String, Numeric
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class Tags(Base):
    """Modèle des tags stockés dans la bdd pour chaque livre
    """
    __tablename__ = "tags"
    tag_id = Column('tag_id', Integer, primary_key=True)
    tag_name = Column('tag_name', String, nullable=False)


class Books(Base):
    """Modèle des livres stockés en bdd
    """
    __tablename__ = "books"
    book_id = Column('book_id', Integer, primary_key=True)
    title = Column('book_title', String, nullable=False)
    original_title = Column('book_original_title', String, nullable=False)
    authors = Column('book_authors', String, nullable=False)
    language_code = Column('book_language_code', String, nullable=False)
    original_publication_year = Column('book_original_publication_year', Integer, nullable=False)
    average_rating = Column('book_average_rating', Numeric(3,2), nullable=False)
    isbn = Column('book_isbn', Integer, nullable=False)
    isbn13 = Column('book_isbn13', Integer)
    ratings_count = Column('book_ratings_count', Integer, nullable=False)
    work_ratings_count = Column('book_work_ratings_count', Integer, nullable=False)
    work_text_reviews_count = Column('book_work_text_reviews_count', Integer, nullable=False)
    ratings_1 = Column('book_ratings_1', Integer, nullable=False)
    ratings_2 = Column('book_ratings_2', Integer, nullable=False)
    ratings_3 = Column('book_ratings_3', Integer, nullable=False)
    ratings_4 = Column('book_ratings_4', Integer, nullable=False)
    ratings_5 = Column('book_ratings_5', Integer, nullable=False)
    image_url = Column('book_image_url', String)
    small_image_url = Column('book_small_image_url', String)
    books_count = Column('book_books_count', Integer)
    work_id = Column('book_work_id', Integer)
    best_book_id = Column('book_best_book_id', Integer)
    goodreads_book_id = Column('book_goodreads_book_id', Integer)
    tags = relationship("Tags", secondary="books_tags", backref="books")
    ratings = relationship("Ratings")
    to_reads = relationship("ToRead")


class BooksTags(Base):
    """Table d'association entre les tags et les livres (many-to-many)
    """
    __tablename__ = "books_tags"
    tag_id = Column('tag_id', ForeignKey('tags.tag_id'), primary_key=True)
    book_id = Column('book_id', ForeignKey('books.book_id'), primary_key=True)


class Ratings(Base):
    """Modèle pour les notes données par chaque utilisateur à un livre
    """
    __tablename__ = "ratings"
    book_id = Column('book_id', ForeignKey('books.book_id'), primary_key=True)
    user_id = Column('user_id', Integer, primary_key=True)
    rating = Column('rt_rating', Integer, nullable=False)


class ToRead(Base):
    """Modèle stockant les livres que veulent lire les utilisateurs
    """
    __tablename__ = "to_read"
    book_id = Column('book_id', ForeignKey('books.book_id'), primary_key=True)
    user_id = Column('user_id', Integer, primary_key=True)
