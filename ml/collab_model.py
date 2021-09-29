import os
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from .ratings_model import RatingsModel

class CollaborativeBasedModel():
    def __init__(self, path: str = "saved_models"):
        self.__path = path
        self.__brute_force = None
    def _fill_na_books(self, books: pd.DataFrame) -> pd.DataFrame:
        """Fonction permettant de remplir les valeurs manquantes par des valeurs vides ou égales à 0 inplace

        Args:
            books (pd.DataFrame): DataFrame de livres

        Returns:
            pd.DataFrame: Le dataframe remplacé
        """
        books = books.fillna({"book_isbn13": 0})
        books["book_isbn"].fillna("", inplace=True)
        books["book_original_title"].fillna("", inplace=True)
        books["book_original_publication_year"].fillna(0, inplace=True)
        books["book_language_code"].fillna("", inplace=True)
        return books
    
    def _filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # On ne garde que les colonnes qui nous intéresse
        data = data.drop(
            columns=set(data.columns).difference(
                set(["book_id", "book_title", "tags", "user_id", "rt_rating"])
            )
        )
        return data
    
    def _to_tensor(self, data: pd.DataFrame):
        """Transforme un DataFrame avec aucunes valeurs NA en un Tensor Dataset

        Args:
            data (pd.DataFrame): Le dataframe à transformer

        Returns:
            Dataset: A tensor dataset
        """
        return tf.data.Dataset.from_tensor_slices(dict(data))

    def train(self, data: pd.DataFrame, books: pd.DataFrame):
        # On remplit les valeurs manquantes car tensorflow ne peut pas convertir de dataframe avec des NA
        books = self._fill_na_books(books)
        # On limite les colonnes de notre dataframe à seulement celles dont nous avons besoin
        data = self._filter_data(data)
        # On convertit les dataframe pandas en tensor datasets
        data = self._to_tensor(data)
        books = self._to_tensor(books)
        # On mélange le dataset et on le sépare en données d'entraînement et de test
        shuffled = data.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
        train = shuffled.take(80_000)
        test = shuffled.skip(80_000).take(20_000)

        # On crée notre modèle de notations
        ratings_model = RatingsModel(train, books, 1.0, 1.0)

        # On compile le modèle avec une descende de gradient Adagrad et un learning rate de 0.1
        ratings_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

        # On met en cache les données d'entraînement et de test pour de meilleurs performances
        retrieval_cached_ratings_trainset = train.shuffle(100_000).batch(8192).cache()
        retrieval_cached_ratings_testset = test.batch(4096).cache()

        # On entraîne le modèle sur 25 epochs et on récupère les mesures d'entraînement
        ratings_model.fit(
            retrieval_cached_ratings_trainset,
            validation_data=retrieval_cached_ratings_testset,
            validation_freq=1,
            epochs=25,
        )

        # On applique un alogrithme de sélection BruteForce afin de récupérer les meilleurs recommandations pour un utilisateur donné
        self.__brute_force = tfrs.layers.factorized_top_k.BruteForce(
            ratings_model.query_model
        )
        self.__brute_force.index_from_dataset(
            books.batch(128).map(
                lambda book: (book["book_title"], ratings_model.candidate_model(book))
            )
        )

        # on test l'algorithme sur un utilisateur afin de générer les formes du BruteForce
        user_id = 43675
        self.__brute_force(tf.constant([user_id]))
    
    def predict(self, user_id: int):
        """Prédit les livres qu'un utilisateur aimerai lire selon ses notes et les notes d'autres utilisateurs

        Args:
            user_id (int): L'id de l'utilisateur dans le DataFrame

        Returns:
            Tensor Dataset: Un dataset avec le titre des livres en première position et leur données en deuxième position
        """
        return self.__brute_force(tf.constant([user_id]))

    def load(self):
        if not os.path.exists(os.path.join(self.__path, "collab_model")):
            raise ValueError(f"The path doesn't exist, the model need to be generated in this path ({os.path.join(self.__path, 'collab_model')})")
        self.__brute_force = tf.saved_model.load(os.path.join(self.__path, "collab_model"))
    
    def save(self):
        if self.__brute_force is None:
            raise ValueError("BruteForce model is not generated")
        tf.saved_model.save(self.__brute_force, os.path.join(self.__path, "collab_model"))
    
    def train_save(self, data: pd.DataFrame, books: pd.DataFrame):
        self.train(data, books)
        self.save()