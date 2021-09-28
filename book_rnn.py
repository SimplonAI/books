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


def fill_na_books(books: pd.DataFrame):
    """Fonction permettant de remplir les valeurs manquantes par des valeurs vides ou égales à 0 inplace

    Args:
        books (pd.DataFrame): DataFrame de livres
    
    Returns:
        pd.DataFrame: Le dataframe remplacé
    """
    books['book_isbn13'].fillna(0, inplace=True)
    books['book_isbn'].fillna('', inplace=True)
    books['book_original_title'].fillna('', inplace=True)
    books['book_original_publication_year'].fillna(0, inplace=True)
    books['book_language_code'].fillna('', inplace=True)
    return books

class UserModel(tfrs.Model):
    """Modèle de recommendation Query basé sur l'utilisateur
    """
    def __init__(self, data: tf.data.Dataset.TensorSliceDataset):
        super().__init__()

        # Make a Keras IntegerLookup layer as the mapping (lookup)
        user_id_lookup_layer = tf.keras.layers.experimental.preprocessing.IntegerLookup(mask_token=None)

        # IntegerLookup layer is a non-trainable layer and its state (the vocabulary)
        # must be constructed and set before training in a step called "adaptation".
        user_id_lookup_layer.adapt(data.map(lambda x: x['user_id']))
        self.user_embedding = tf.keras.Sequential([
            user_id_lookup_layer,
            tf.keras.layers.Embedding(user_id_lookup_layer.vocabulary_size(), 48),
        ])
    
    def call(self, user_id):
        return self.user_embedding(user_id)
    


class BookModel(tfrs.Model):
    """Modèle de recommendation Candidat basé sur les livres (id + titres + tags)
    """
    def __init__(self, data, max_tokens=10_000):
        super().__init__()
        # Make a Keras IntegerLookup layer as the mapping (lookup)
        book_id_lookup_layer = tf.keras.layers.experimental.preprocessing.IntegerLookup(mask_token=None)

        # IntegerLookup layer is a non-trainable layer and its state (the vocabulary)
        # must be constructed and set before training in a step called "adaptation".
        book_id_lookup_layer.adapt(data.map(lambda x: x['book_id']))

        book_title_lookup_layer = tf.keras.layers.experimental.preprocessing.TextVectorization()
        book_title_lookup_layer.adapt(data.map(lambda x: x['book_title']))
        book_tags_lookup_layer = tf.keras.layers.experimental.preprocessing.TextVectorization()
        book_tags_lookup_layer.adapt(data.map(lambda x: x['tags']))

        self.book_embedding = tf.keras.Sequential([
            book_id_lookup_layer,
            tf.keras.layers.Embedding(book_id_lookup_layer.vocabulary_size(), 16),
        ])
        self.book_title_embedding = tf.keras.Sequential([
            book_title_lookup_layer,
            tf.keras.layers.Embedding(max_tokens, 16, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])
        self.book_tags_embedding = tf.keras.Sequential([
            book_tags_lookup_layer,
            tf.keras.layers.Embedding(max_tokens, 16, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])
    
    def call(self, inputs):
        return tf.concat([
            self.book_embedding(inputs["book_id"]),
            self.book_title_embedding(inputs["book_title"]),
            self.book_tags_embedding(inputs["tags"]),
        ], axis=1)

class QueryModel(tf.keras.Model):
  """Model for encoding user queries."""

  def __init__(self, data, layer_sizes):
    """Model for encoding user queries.

    Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super().__init__()

    # We first use the user model for generating embeddings.
    self.embedding_model = UserModel(data)

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    # No activation for the last layer.
    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))
    
  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)

class CandidateModel(tf.keras.Model):
  """Model for encoding movies."""

  def __init__(self, data, layer_sizes):
    """Model for encoding movies.

    Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super().__init__()

    self.embedding_model = BookModel(data)

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    # No activation for the last layer.
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

        self.query_model = QueryModel(data, layer_sizes)
        self.candidate_model = CandidateModel(data, layer_sizes)

        # A small model to take in user and movie embeddings and predict ratings.
        # We can make this as complicated as we want as long as we output a scalar
        # as our prediction.
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=books.batch(128).map(self.candidate_model)
            )
        )
        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight
    def call(self, features: dict[str, tf.Tensor]) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.query_model(features["user_id"])
        # And pick out the book features and pass them into the book model.
        book_embeddings = self.candidate_model({
            'book_id': features["book_id"],
            'book_title': features["book_title"],
            'tags': features["tags"],
        })

        return (
            user_embeddings,
            book_embeddings,
            # We apply the multi-layered rating model to a concatentation of
            # user and book embeddings.
            self.rating_model(
                tf.concat([user_embeddings, book_embeddings], axis=1)
            ),
        )
    def compute_loss(self, features, training=False) -> tf.Tensor:
        ratings = features.pop("rt_rating")

        user_embeddings, book_embeddings, rating_predictions = self(features)

        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, book_embeddings)

        # And combine them using the loss weights.
        return (self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss)


def plot_history(history):
    """Fonction qui permet de plotter les courbes de précision selon l'epoch et de perte

    Args:
        history: l'historique des métriques calculées par epoch
    """
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model losses during training")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    # Plot changes in model accuracy during training
    plt.plot(history.history["factorized_top_k/top_100_categorical_accuracy"])
    plt.plot(history.history["val_factorized_top_k/top_100_categorical_accuracy"])
    plt.title("Model accuracies during training")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

async def main():
    data, books = await DataManager(db, keep_data=False).data
    data = data.drop(columns=set(data.columns).difference(set(["book_id", "book_title", "tags", "user_id", "rt_rating"])))

    fill_na_books(books)

    data = tf.data.Dataset.from_tensor_slices(dict(data))
    books = tf.data.Dataset.from_tensor_slices(dict(books))
    data = data.map(lambda x: {
        "book_id": x["book_id"],
        "book_title": x["book_title"],
        "tags": x["tags"],
        "user_id": x["user_id"],
        "rt_rating": x["rt_rating"]
    })

    tf.random.set_seed(42)
    shuffled = data.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)
    
    ratings_model = RatingsModel(train, books, 1.0, 1.0)
    optimizer_step_size = 0.1
    ratings_model.compile(
        optimizer=tf.keras.optimizers.Adagrad(
            learning_rate=optimizer_step_size
        )
    )

    retrieval_cached_ratings_trainset = train.shuffle(100_000).batch(8192).cache()
    retrieval_cached_ratings_testset = test.batch(4096).cache()
    
    num_epochs = 3
    history = ratings_model.fit(
        retrieval_cached_ratings_trainset,
        validation_data=retrieval_cached_ratings_testset,
        validation_freq=1,
        epochs=num_epochs
    )

    plot_history(history)

    brute_force_layer = tfrs.layers.factorized_top_k.BruteForce(
        ratings_model.query_model
    )
    brute_force_layer.index_from_dataset(
        books.batch(128).map(lambda book: (book["book_title"], ratings_model.candidate_model(book)))
    )


    user_id = 43675
    afinity_scores, movie_ids = brute_force_layer(
        tf.constant([user_id])
    )

    print(f"Recommendations for user {user_id} using BruteForce: {movie_ids[0, :5]}")

    scann_layer = tfrs.layers.factorized_top_k.ScaNN(
        ratings_model.query_model
    )

    scann_layer.index_from_dataset(
        books.batch(128).map(lambda book: (book["book_title"], ratings_model.candidate_model(book)))
    )

    afinity_scores, movie_ids = scann_layer(
        tf.constant([user_id])
    )

    print(f"Recommendations for user {user_id} using ScaNN: {movie_ids[0, :5]}")



look = asyncio.get_event_loop()
look.run_until_complete(main())