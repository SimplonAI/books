from __future__ import annotations
import pandas as pd
from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem
from db.models import Books, BooksTags, Ratings, Tags, ToRead

def add_prefix(data: pd.DataFrame, prefix: str, filter: (list | str) = []):
    if isinstance(filter, str):
        filter = [filter]
    return data.rename(
        columns={
            col: prefix + col if col not in filter else col for col in data.columns
        }
    )


def insert_data(db):
    # Chargement des données csv
    ratings = pd.read_csv("../ratings.csv")
    books = pd.read_csv("../books.csv")
    tags = pd.read_csv("../tags.csv")
    book_tags = pd.read_csv("../book_tags.csv")
    to_read = pd.read_csv("../to_read.csv")

    books.isbn13 = books.isbn13.astype("Int64")
    books = add_prefix(books, "book_", "book_id")
    book_tags.drop_duplicates(["goodreads_book_id", "tag_id"], inplace=True)
    ratings = add_prefix(ratings, "rt_", ["book_id", "user_id"])

    with db.begin() as conn:
        Books.insert_df(books, conn)
        print("[1/5] Livres insérés")
        Ratings.insert_df(ratings, conn)
        print("[2/5] Notes insérés")
        Tags.insert_df(tags, conn)
        print("[3/5] Tags insérés")
        ToRead.insert_df(to_read, conn)
        print("[4/5] Liste de lecture inséré")
        BooksTags.insert_df(book_tags, conn)
        print("[5/5] Relation Livres-Tags inséré")
    print("Tout a été inséré !")


def add_data(file):
    data = pd.read_csv(file)
    print("look for data with API")


def add_data_menu():
    file = ""
    while file == "":
        file = input("Chemin vers le fichier csv (q pour quitter) : ")
    if file != "q":
        add_data(file)


if __name__ == "__main__":
    from db import db
    menu = ConsoleMenu()
    menu.append_item(FunctionItem("Insertion initiale des données csv", lambda: insert_data(db)))
    menu.append_item(FunctionItem("Importation de données partielles", add_data_menu))
    menu.show()
