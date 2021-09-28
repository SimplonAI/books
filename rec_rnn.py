import asyncio
import collections
from enum import Enum, auto
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from data_manager import DataManager
tf.compat.v1.disable_eager_execution()

class ComputeMethodEnum(Enum):
    DOT = auto()
    COSINE = auto()

class RecFactory():
    """Classe permettant de construire un modèle de recommandation
    """
    def __init__(self, ratings: pd.DataFrame, logger=None):
        if logger is not None:
            assert callable(getattr(logger, "debug", None)), "Votre classe Logger n'implémente pas une fonction debug"
        self._logger = logger
        print(ratings)
        self._reindex(ratings, inplace=True)
        print(ratings)
        self.ratings = ratings
       
    def _reindex(self, ratings: pd.DataFrame, inplace=False):
        """Fonction permettant de réindexer les user_id et les book_id de manière continu en partant de 0, nécessaire pour tensorflow/keras

        Args:
            ratings (pd.DataFrame): Le DataFrame à réindexer
            inplace (bool): permet de remplacer en ligne le dataframe passé en argument

        Returns:
            DataFrame: Retourne le DataFrame réindexé
        """
        # On récupère les id unique et on les range afin de garder le même ordre une fois réindexé
        user_ids = ratings["user_id"].unique()
        user_ids.sort()
        book_ids = ratings["book_id"].unique()
        book_ids.sort()
        # On stocke dans un hashmap les id réindexé et leur valeur d'origine
        self._user_ids_idx = dict(zip(range(0, len(user_ids)), user_ids.tolist()))
        self._book_ids_idx = dict(zip(range(0, len(book_ids)), book_ids.tolist()))
        # Si on ne remplace pas inplace, on crée une copie du dataframe (attention à la ram !)
        if not inplace:
            rt = ratings.copy()
        else:
            rt = ratings
        # On réindexe les ids 
        rt["user_id"] = rt["user_id"].map(dict(zip(self._user_ids_idx.values(), self._user_ids_idx.keys()))).astype(int)
        rt["book_id"] = rt["book_id"].map(dict(zip(self._book_ids_idx.values(), self._book_ids_idx.keys()))).astype(int)
        return rt

    def get_user_id(self, user_id: int) -> int:
        """Permet de récupérer l'id de l'utilisateur original du DataFrame avant sa réindexation 

        Args:
            user_id (int): L'id de l'utilisateur dans le DataFrame réindexé

        Returns:
            int: L'id de l'utilisateur dans le DataFrame original avant sa réindexation
        """
        if self._user_ids_idx is not None:
            return self._user_ids_idx.get(user_id)
        return None

    def get_book_id(self, book_id: int) -> int:
        """Permet de récupérer l'id du livre du DataFrame original avant sa réindexation

        Args:
            book_id (int): L'id du livre dans le DataFrame réindexé 

        Returns:
            int: L'id du livre dans le DataFrame original avant sa réindexation
        """
        if self._book_ids_idx is not None:
            return self._book_ids_idx.get(book_id)
        return None

    def _log(self, msg: str, *args, **kwargs):
        """Fonction permettant de logger des messages d'informations lors du développement

        Args:
            msg (str): Un message à logger
        """
        if self._logger is not None:
            self._logger.debug(msg, *args, **kwargs)

    def build_rating_sparse_tensor(self, ratings_df: pd.DataFrame):
        """Fonction qui construit un sparse tensor avec `user_id`, `book_id` et `rt_rating` du DataFrame `ratings_df`

        Args:
            ratings_df (pd.DataFrame): un pd.DataFrame avec des colonnes `user_id`, `book_id` et `rt_rating`.

        Returns:
            tf.SparseTensor: Le sparse tensor construit
        """
        indices = ratings_df[['user_id', 'book_id']].values
        values = ratings_df['rt_rating'].values
        return tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=[self.ratings["user_id"].drop_duplicates().shape[0], self.ratings["book_id"].drop_duplicates().shape[0]])
    def sparse_mean_square_error(self, sparse_ratings, user_embeddings, book_embeddings):
        """Calcul l'erreur quadratique moyenne de la sparse matrice
 
        Args:
            sparse_ratings: Un SparseTensor de la matrice des notes de taille [N, M]
            user_embeddings: Un dense Tensor U de taille [N, k] où k est la dimension de l'embedding, tel que U_i est l'embedding de l'utilisateur i.
            book_embeddings: Un dense Tensor V de taille [M, k] où k est ma dimension de l'embedding, tel que V_j est l'embedding du livre j.

        Returns:
            Un Tensor scalaire représentant le MSE entre les vrai notes et les prédictions du modèle.
        """
        predictions = tf.reduce_sum(
            tf.gather(user_embeddings, sparse_ratings.indices[:, 0]) *
            tf.gather(book_embeddings, sparse_ratings.indices[:, 1]),
            axis=1)
        loss = tf.losses.mean_squared_error(sparse_ratings.values, predictions)
        return loss

    def build_model(self, cls, embedding_dim=3, init_stddev=1., ):
         # Split the ratings DataFrame into train and test.
        train_ratings, test_ratings = train_test_split(self.ratings)
        # SparseTensor representation of the train and test datasets.
        A_train = self.build_rating_sparse_tensor(train_ratings)
        A_test = self.build_rating_sparse_tensor(test_ratings)
        # Initialize the embeddings using a normal distribution.
        U = tf.Variable(tf.random_normal(
            [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
        V = tf.Variable(tf.random_normal(
            [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))
        train_loss = self.sparse_mean_square_error(A_train, U, V)
        test_loss = self.sparse_mean_square_error(A_test, U, V)
        metrics = {
            'train_error': train_loss,
            'test_error': test_loss
        }
        embeddings = {
            "user_id": U,
            "movie_id": V
        }
        return cls(embeddings, train_loss, [metrics])
    def compute_scores(self, query_embedding, item_embeddings, measure: ComputeMethodEnum=ComputeMethodEnum.DOT):
        """Computes the scores of the candidates given a query.
        Args:
            query_embedding: a vector of shape [k], representing the query embedding.
            item_embeddings: a matrix of shape [N, k], such that row i is the embedding
            of item i.
            measure: a string specifying the similarity measure to be used. Can be
            either DOT or COSINE.
        Returns:
            scores: a vector of shape [N], such that scores[i] is the score of item i.
        """
        u = query_embedding
        V = item_embeddings
        if measure == ComputeMethodEnum.COSINE:
            V = V / np.linalg.norm(V, axis=1, keepdims=True)
            u = u / np.linalg.norm(u)
        scores = u.dot(V.T)
        return scores

    def user_recommendations(self, model, measure: ComputeMethodEnum=ComputeMethodEnum.DOT, exclude_rated=False, user_id: int=None, k=6):
        scores = self.compute_scores(
            model.embeddings["user_id"][943], model.embeddings["movie_id"], measure)
        score_key = measure + ' score'
        books = self.ratings[self.ratings["book_id"].drop_duplicates().index]
        df = pd.DataFrame({
            score_key: list(scores),
            'movie_id': books['book_id'],
            'titles': books['book_title'],
            'genres': books['tags'],
        })
        if exclude_rated and user_id is not None:
            # remove movies that are already rated
            user_id = dict(zip(self._user_ids_idx.values(), self._user_ids_idx.keys()))[user_id]
            rated_books = self.ratings[self.ratings.user_id == user_id]["book_id"].values
            df = df[df.movie_id.apply(lambda movie_id: movie_id not in rated_books)]
        return df.sort_values([score_key], ascending=False).head(k)  

class CFModel(object):
  """Une classe mettant en place un modèle de recommandation collaboratif"""
  def __init__(self, embedding_vars, loss, metrics=None):
    """Initialise la classe 
    Args:
      embedding_vars: Un dictionnaire de tf.Variables.
      loss: Un Tensor float. La perte à optimiser.
      metrics: Une liste optionnelle de dictionnaire de Tensors. Les mesures de chaque 
      dictionnaire seront plot dans une figure séparée durant l'entraînement.
    """
    self._embedding_vars = embedding_vars
    self._loss = loss
    self._metrics = metrics
    self._embeddings = {k: None for k in embedding_vars}
    self._session = None

  @property
  def embeddings(self):
    """Le dictionnaire d'embeddings"""
    return self._embeddings

  def train(self, num_iterations=100, learning_rate=1.0, plot_results=True,
            optimizer=tf.train.GradientDescentOptimizer):
    """Fonction qui entraine le modèle
    
    Args:
      iterations (int): Nombre d'itérations à exécuter.
      learning_rate (float): Taux d'apprentissage.
      plot_results (bool): Si vrai, plot les résultats à la fin de l'entraînement.
      optimizer: L'algorithme d'optimisation à utiliser. GradientDescentOptimizer par défaut.
    
    Returns:
      dict: Le dictionnaire de mesures calculé à la fin de l'entraînement.'
    """
    with self._loss.graph.as_default():
      opt = optimizer(learning_rate)
      train_op = opt.minimize(self._loss)
      local_init_op = tf.group(
          tf.variables_initializer(opt.variables()),
          tf.local_variables_initializer())
      if self._session is None:
        self._session = tf.Session()
        with self._session.as_default():
          self._session.run(tf.global_variables_initializer())
          self._session.run(tf.tables_initializer())
          tf.train.start_queue_runners()

    with self._session.as_default():
      local_init_op.run()
      iterations = []
      metrics = self._metrics or ({},)
      metrics_vals = [collections.defaultdict(list) for _ in self._metrics]

      # Train and append results.
      for i in range(num_iterations + 1):
        _, results = self._session.run((train_op, metrics))
        if (i % 10 == 0) or i == num_iterations:
          print("\r iteration %d: " % i + ", ".join(
                ["%s=%f" % (k, v) for r in results for k, v in r.items()]),
                end='')
          iterations.append(i)
          for metric_val, result in zip(metrics_vals, results):
            for k, v in result.items():
              metric_val[k].append(v)

      for k, v in self._embedding_vars.items():
        self._embeddings[k] = v.eval()

      if plot_results:
        # Plot the metrics.
        num_subplots = len(metrics)+1
        fig = plt.figure()
        fig.set_size_inches(num_subplots*10, 8)
        for i, metric_vals in enumerate(metrics_vals):
          ax = fig.add_subplot(1, num_subplots, i+1)
          for k, v in metric_vals.items():
            ax.plot(iterations, v, label=k)
          ax.set_xlim([1, num_iterations])
          ax.legend()
      return results

async def main():
    """Fonction qui s'exécuter lors du lancement du script python
    """
    from db.db import db
    data_manager = DataManager(db, keep_data=False)
    data, _ = await data_manager.data
    rec_rnn = RecFactory(data)
    model = rec_rnn.build_model(CFModel)
    model.train(num_iterations=2000, learning_rate=10.)
    plt.show()    

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
