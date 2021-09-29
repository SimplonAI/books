import tensorflow as tf
import tensorflow_recommenders as tfrs

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