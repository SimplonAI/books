from __future__ import annotations
import asyncio
from typing import Tuple
from data_manager import DataManager
from db.db import db
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import matplotlib.pyplot as plt
import os


def fill_na_books(books: pd.DataFrame):
    """Fonction permettant de remplir les valeurs manquantes par des valeurs vides ou égales à 0 inplace

    Args:
        books (pd.DataFrame): DataFrame de livres

    Returns:
        pd.DataFrame: Le dataframe remplacé
    """
    books["book_isbn13"].fillna(0, inplace=True)
    books["book_isbn"].fillna("", inplace=True)
    books["book_original_title"].fillna("", inplace=True)
    books["book_original_publication_year"].fillna(0, inplace=True)
    books["book_language_code"].fillna("", inplace=True)
    return books


class UserModel(tfrs.Model):
    """Modèle de recommendation Query basé sur l'utilisateur"""

    def __init__(self, data):
        super().__init__()

        # On crée un layer IntegerLookup afin de faire automatiquement le mapping (id continue de 0 à n)
        user_id_lookup_layer = tf.keras.layers.experimental.preprocessing.IntegerLookup(
            mask_token=None
        )

        # Le layer IntegerLookup est un layer qui ne s'entraîne pas, son état (le vocabulaire) doit être construit et mis en place avant l'entraînement
        user_id_lookup_layer.adapt(data.map(lambda x: x["user_id"]))

        # On crée l'embedding pour les utilisateurs (id)
        self.user_embedding = tf.keras.Sequential(
            [
                user_id_lookup_layer,
                tf.keras.layers.Embedding(user_id_lookup_layer.vocabulary_size(), 48),
            ]
        )

    def call(self, user_id):
        return self.user_embedding(user_id)


class BookModel(tfrs.Model):
    """Modèle de recommendation Candidat basé sur les livres (id + titres + tags)"""

    def __init__(self, data, max_tokens=10_000):
        super().__init__()
        # On crée un layer IntegerLookup afin de faire automatiquement le mapping (id continue de 0 à n)
        book_id_lookup_layer = tf.keras.layers.experimental.preprocessing.IntegerLookup(
            mask_token=None
        )

        # Le layer IntegerLookup est un layer qui ne s'entraîne pas, son état (le vocabulaire) doit être construit et mis en place avant l'entraînement
        book_id_lookup_layer.adapt(data.map(lambda x: x["book_id"]))

        # On crée notre vecteur de mots pour les titres (même comportement qu'un IntegerLookup)
        book_title_lookup_layer = (
            tf.keras.layers.experimental.preprocessing.TextVectorization()
        )
        book_title_lookup_layer.adapt(data.map(lambda x: x["book_title"]))

        # On crée notre vecteur de mots pour les tags (même comportement qu'un IntegerLookup)
        book_tags_lookup_layer = (
            tf.keras.layers.experimental.preprocessing.TextVectorization()
        )
        book_tags_lookup_layer.adapt(data.map(lambda x: x["tags"]))

        # On crée les embedding pour les ids, les titres et les tags
        self.book_embedding = tf.keras.Sequential(
            [
                book_id_lookup_layer,
                tf.keras.layers.Embedding(book_id_lookup_layer.vocabulary_size(), 16),
            ]
        )
        self.book_title_embedding = tf.keras.Sequential(
            [
                book_title_lookup_layer,
                tf.keras.layers.Embedding(max_tokens, 16, mask_zero=True),
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )
        self.book_tags_embedding = tf.keras.Sequential(
            [
                book_tags_lookup_layer,
                tf.keras.layers.Embedding(max_tokens, 16, mask_zero=True),
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )

    def call(self, inputs):
        return tf.concat(
            [
                self.book_embedding(inputs["book_id"]),
                self.book_title_embedding(inputs["book_title"]),
                self.book_tags_embedding(inputs["tags"]),
            ],
            axis=1,
        )


class QueryModel(tf.keras.Model):
    """Modèle pour encoder les requêtes utilisateurs."""

    def __init__(self, data, layer_sizes):
        """Modèle pour encoder les requêtes utilisateurs.

        Args:
            data: Tensor Dataset sur lequel le modèle doit faire sa requête
            layer_sizes (list[int]): Une liste d'int où l'élément à la position i représente the nombre de neurones
            le layer i contient.
        """
        super().__init__()

        # On utilise le UserModel afin de générer les embeddings.
        self.embedding_model = UserModel(data)

        # On construit nos layers
        self.dense_layers = tf.keras.Sequential()

        # Pour chaque layer, on intercale une fonction d'activation (Relu ici) sauf pour le dernier layer
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # Pas d'activation pour le dernier layer
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class CandidateModel(tf.keras.Model):
    """Model pour encoder les livres."""

    def __init__(self, data, layer_sizes):
        """Model pour encoder les livres.

        Args:
            data: Tensor Dataset sur lequel le modèle doit faire sa requête
            layer_sizes (list[int]): Une liste d'int où l'élément à la position i représente the nombre de neurones
            le layer i contient.
        """
        super().__init__()

        # On utilise le BookModel afin de générer les embeddings.
        self.embedding_model = BookModel(data)

        # On construit nos layers
        self.dense_layers = tf.keras.Sequential()

        # Pour chaque layer, on intercale une fonction d'activation (Relu ici) sauf pour le dernier layer
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # Pas d'activation pour le dernier layer
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class RatingsModel(tfrs.Model):
    """Modèle de recommendation TensorFlow regroupant les modèles de query et candidat afin de récupérer les livres selon
    les features implicites
    """

    def __init__(self, data, books, rating_weight, retrieval_weight, layer_sizes=[32]):
        super().__init__()

        # Les modèles de requètes et de candidats
        self.query_model = QueryModel(data, layer_sizes)
        self.candidate_model = CandidateModel(data, layer_sizes)

        # Modèle qui prend les embeddings utilisateurs et de films et prédit les notes.
        self.rating_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

        # La tache de notation comparant les vrai notes avec celles prédites
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

        # La tache de récupération comparant les livres qu'a lu l'utilisateur avec ceux prédits
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=books.batch(128).map(self.candidate_model)
            )
        )

        # Les poids qu'on attribue à chaque modèle (notation et récupération)
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def call(self, features: dict[str, tf.Tensor]) -> tf.Tensor:
        # On choisit les features utilisateurs et on les donne au modèle query (utilisateur).
        user_embeddings = self.query_model(features["user_id"])
        # On choisit les features des livres et on les donne au modèle candidat (livre).
        book_embeddings = self.candidate_model(
            {
                "book_id": features["book_id"],
                "book_title": features["book_title"],
                "tags": features["tags"],
            }
        )

        return (
            user_embeddings,
            book_embeddings,
            # On applique le modèle à multi couches de notation à une concaténation des embeddings utilisateurs et livres
            self.rating_model(tf.concat([user_embeddings, book_embeddings], axis=1)),
        )

    def compute_loss(self, features, training=False) -> tf.Tensor:
        ratings = features.pop("rt_rating")

        user_embeddings, book_embeddings, rating_predictions = self(features)

        # On calcule la perte pour chaque tâche.
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, book_embeddings)

        # Et on retourne leur combinaison avec les poids prédéfinis à l'initialisation.
        return self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss


def plot_history(history):
    """Fonction qui permet de plotter les courbes de précision selon l'epoch et de perte

    Args:
        history: l'historique des métriques calculées par epoch
    """
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Perte du modèle pendant l'entraînement")
    plt.xlabel("epoch")
    plt.ylabel("perte")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    plt.plot(history.history["factorized_top_k/top_100_categorical_accuracy"])
    plt.plot(history.history["val_factorized_top_k/top_100_categorical_accuracy"])
    plt.title("Précision du modèle pendant l'entraînement")
    plt.xlabel("epoch")
    plt.ylabel("précision")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()


async def main():
    # On récupère les données de la BDD (sans mise en cache)
    data, books = await DataManager(db, keep_data=False).data
    # On ne garde que les colonnes qui nous intéresse
    data = data.drop(
        columns=set(data.columns).difference(
            set(["book_id", "book_title", "tags", "user_id", "rt_rating"])
        )
    )

    # On remplit les valeurs manquantes car tensorflow ne peut pas convertir de dataframe avec des NA
    fill_na_books(books)

    # On convertit les dataframe pandas en tensor datasets
    data = tf.data.Dataset.from_tensor_slices(dict(data))
    books = tf.data.Dataset.from_tensor_slices(dict(books))

    data = data.map(
        lambda x: {
            "book_id": x["book_id"],
            "book_title": x["book_title"],
            "tags": x["tags"],
            "user_id": x["user_id"],
            "rt_rating": x["rt_rating"],
        }
    )

    tf.random.set_seed(42)
    # On mélange le dataset et on le sépare en données d'entraînement et de test
    shuffled = data.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)
    
    # On crée notre modèle de notations
    ratings_model = RatingsModel(train, books, 1.0, 1.0)

    # On compile le modèle avec une descende de gradient Adagrad et un learning rate de 0.1
    ratings_model.compile(
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1)
    )

    # On met en cache les données d'entraînement et de test pour de meilleurs performances
    retrieval_cached_ratings_trainset = train.shuffle(100_000).batch(8192).cache()
    retrieval_cached_ratings_testset = test.batch(4096).cache()

    # On entraîne le modèle sur 25 epochs et on récupère les mesures d'entraînement 
    history = ratings_model.fit(
        retrieval_cached_ratings_trainset,
        validation_data=retrieval_cached_ratings_testset,
        validation_freq=1,
        epochs=25,
    )

    # on plot les mesures
    plot_history(history)

    # On applique un alogrithme de sélection BruteForce afin de récupérer les meilleurs recommandations pour un utilisateur donné
    brute_force_layer = tfrs.layers.factorized_top_k.BruteForce(
        ratings_model.query_model
    )
    brute_force_layer.index_from_dataset(
        books.batch(128).map(
            lambda book: (book["book_title"], ratings_model.candidate_model(book))
        )
    )

    # on test l'algorithme sur un utilisateur afin de générer les formes du BruteForce
    user_id = 43675
    afinity_scores, movie_ids = brute_force_layer(tf.constant([user_id]))

    print(f"Recommandations pour l'utilisateur {user_id} en utilisant BruteForce : {movie_ids[0, :5]}")

    # On sauvegarde notre modèle final
    path = os.path.join("saved_models", "model_rnn")
    tf.saved_model.save(brute_force_layer, path)


look = asyncio.get_event_loop()
look.run_until_complete(main())
