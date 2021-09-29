import tensorflow as tf
import tensorflow_recommenders as tfrs

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