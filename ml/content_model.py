import os
from joblib import dump, load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedModel:
    def __init__(
        self, books: pd.DataFrame, path: str = "saved_models"
    ):
        self.__path = path
        self.books = books

    def load(self):
        """Charge le modèle enregistré
        """
        self.__cosine_sim = load(os.path.join(self.__path, "content_based_saved.joblib"))

    def save(self):
        """Enregistre le modèle entraîné
        """
        dump(self.__cosine_sim, os.path.join(self.__path, "content_based_saved.joblib"))

    def predict(self, book: str, slices: list[tuple[int, int]]=[(0,5), (45,50)]):
        """Prédit 10 livres similaires à celui donné en argument. 5 les plus similaires et les 5 derniers du top 50.

        Args:
            book (str): Le nom du livre que l'on doit comparer
            size (list[(int, int)]): Slices de résultats à retourner

        Returns:
            pd.DataFrame: Retourne un dataframe contenant 10 livres similaires
        """
        idx = self.books[book["book_title"] == book].index[0]
        # Obtenir l'index du livre qui correspond au titre
        similitude_scores = list(enumerate(self.__cosine_sim[idx]))
        # Obtenir les scores de similarité
        similitude_scores = sorted(similitude_scores, key=lambda x: x[1], reverse=True)
        # Trier les livres en fonction des scores de similarité
        filtered_scores = []
        for _slice in slices:
            start, end = _slice
            filtered_scores += similitude_scores[start:end]
        # Obtenir les scores de 10 livres similaires
        talk_indices = [i[0] for i in filtered_scores]
        # Obtenir les index des livres
        return self.books["title"].iloc[talk_indices]

    def train(self):
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.books["tags"])
        # Contruire une matrice
        self.__cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def train_save(self):
        self.train()
        self.save()