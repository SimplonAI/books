import tensorflow as tf
import tensorflow_recommenders as tfrs
from .query_model import QueryModel
from .candidate_model import CandidateModel

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
