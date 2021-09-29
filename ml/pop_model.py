import pandas as pd

class PopularityBasedModel():
    def __init__(self, books: pd.DataFrame):
        self.books = books
    def predict(self, slices: list[tuple[int, int]]=[(0,5), (45,50)]) -> pd.DataFrame:
        # On récupère les livres ordonnées selon le nombre de vote
        sorted_books = self.books.sort_values("book_ratings_count", ascending=False)
        return pd.concat([sorted_books.iloc[start:end] for (start, end) in slices])
