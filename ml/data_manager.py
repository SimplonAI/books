import asyncio
import re
from threading import Lock
import pandas as pd


class DataManager:
    """Classe permettant de charger les données une seule fois, quel que soit le nombre de thread (thread-safe singleton).
    Il est ainsi possible d'économiser de la mémoire.
    """

    _instance = None
    _lockInstance: Lock = Lock()
    _lockData: Lock = Lock()
    _keepData: bool = False
    _lockKeepData: Lock = Lock()

    def __new__(cls, db, keep_data: bool = False, *args, **kwargs):
        with cls._lockInstance:
            if cls._instance is None:
                cls._instance = super().__new__(cls, *args, **kwargs)
                cls._instance.keep_data = keep_data
                if keep_data:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(cls._instance._load(db))
        return cls._instance

    @property
    def keep_data(self) -> bool:
        with self._lockKeepData:
            return self._keepData

    @keep_data.setter
    def keep_data(self, data: bool) -> None:
        with self._lockKeepData:
            self._keepData = data

    async def _load(self, db):
        """Chargement des données à partir d'une connexion à une base de données

        Args:
            db (Any): Connexion à une base de données
        """
        loop = asyncio.get_event_loop()
        self.__reg = self._make_filter_regexp()
        data = await loop.run_in_executor(
            None,
            pd.read_sql,
            "SELECT b.book_id, b.book_goodreads_book_id, b.book_title, b.book_original_title, b.book_original_publication_year, b.book_average_rating, b.book_image_url, rt.rt_rating, rt.user_id FROM books b INNER JOIN ratings rt ON rt.book_id = b.book_id",
            db,
        )

        books = await loop.run_in_executor(
            None, pd.read_sql, "SELECT * FROM books", db
        )
        tags = await loop.run_in_executor(
            None, pd.read_sql, "SELECT * FROM tags_joined", db
        )
        tags = await loop.run_in_executor(
            None, pd.read_sql, "SELECT * FROM tags_joined", db
        )
        tags = (
            tags[tags["tag_name"].apply(self.filter_tags)].reset_index(drop=True).copy()
        )
        tags["tags"] = tags.groupby("goodreads_book_id")["tag_name"].transform(" ".join)
        tags.drop(["tag_id", "tag_name", "tag_rank", "count"], axis=1, inplace=True)
        tags.drop_duplicates(inplace=True)
        data = data.merge(
            tags,
            how="left",
            left_on="book_goodreads_book_id",
            right_on="goodreads_book_id",
        )
        books = books.merge(
            tags,
            how="left",
            left_on="book_goodreads_book_id",
            right_on="goodreads_book_id",
        )
        if self.keep_data:
            with self._lockData:
                self.__data = data
                self.books = books
        return data, books

    @property
    async def data(self):
        """DataFrame des données chargées depuis la BDD et transformées

        Returns:
            DataFrame: Notes des livres par utilisateurs
        """
        if not self.keep_data:
            return await self._load()
        with self._lockData:
            if self.__data is None:
                raise ValueError("You need to load the data before getting it")
            return self.__data.copy(), self.books.copy()

    def _make_filter_regexp(self, path="exclude_tags.txt"):
        """Retourne une RegExp permettant de filtrer tous les tags depuis un fichier les listant.

        Args:
            path (str): Chemin vers le fichier contenant pour chaque ligne un tag à exclure

        Returns:
            Pattern[str]: RegExp comprenant tous les tags à exclure
        """
        with open(path) as f:
            exclude_tags = f.read().splitlines()
        return re.compile("(?:" + "|".join(exclude_tags) + ")")

    def filter_tags(self, row: str):
        if not row.replace("-", "").isalpha():
            return False
        return self.__reg.search(row) == None
