import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import re
from models import Books, BooksTags, Ratings, Tags, ToRead


def insert_data():
    # Chargement des données csv
    ratings = pd.read_csv("ratings.csv")
    books = pd.read_csv("books.csv")
    tags = pd.read_csv("tags.csv")
    book_tags = pd.read_csv("book_tags.csv")
    to_read = pd.read_csv("to_read.csv")

    Ratings.insert_df(ratings)
    Books.insert_df(ratings)
    Tags.insert_df(ratings)
    ToRead.insert_df(ratings)
    BooksTags.insert_df(ratings)