#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# Import des librairies
#%%
data = pd.read_csv(
    "data.csv",
    index_col=0,
    dtype={
        "user_id": "int64",
        "book_id": "int64",
        "rating": "int64",
        "goodreads_book_id": "int64",
        "best_book_id": "int64",
        "work_id": "int64",
        "books_count": "int64",
        "isbn": "object",
        "isbn13": "float64",
        "authors": "object",
        "original_publication_year": "float64",
        "original_title": "object",
        "title": "object",
        "language_code": "object",
        "average_rating": "float64",
        "ratings_count": "int64",
        "work_ratings_count": "int64",
        "work_text_reviews_count": "int64",
        "ratings_1": "int64",
        "ratings_2": "int64",
        "ratings_3": "int64",
        "ratings_4": "int64",
        "ratings_5": "int64",
        "image_url": "object",
        "small_image_url": "object",
        "to_read": "float64",
        "tags": "object",
    },
)
# Lecture du fichier csv regroupant tous les csv nettoyés
#%%
data.drop_duplicates(subset=["title"], inplace=True)
# Après étude la plupart des livres sont en multiples exemplaires
# s'agissant d'un gros csv très long à charger il faut enlever
# les titres en doubles.
#%%
data.reset_index(inplace=True)
# Cela permet de reindexer
#%%
data.head()
#%%
def recommandations(title, cosine_sim, indices):

    idx = indices[title]
    # Obtenir l'index du film qui correspond au titre
    similitude_scores = list(enumerate(cosine_sim[idx]))
    # Obtenir les scores de similarité
    similitude_scores = sorted(similitude_scores, key=lambda x: x[1], reverse=True)
    # Trier les films en fonction des scores de similarité
    similitude_scores = similitude_scores[0:5] + similitude_scores[40:45]
    # Obtenir les scores des 3 films les plus similaires
    talk_indices = [i[0] for i in similitude_scores]
    # Obtenir les index des films
    return data["title"].iloc[talk_indices]


# Renvoie les films les plus similaires
#%%
indices = pd.Series(data.index, index=data["title"]).drop_duplicates()
# renvoie une serie avec un index et le titre enleve les doublons
stock = data["tags"]
# Tags est stocké dans transcript
#%%
tfidf = TfidfVectorizer()
#%%
tfidf_matrix = tfidf.fit_transform(stock)
# Contruire une matrice
#%%
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Génére une matrice de similarité cosinus
#%%
cosine_sim
#%%
print(recommandations("The Nix", cosine_sim, indices))
# Contenu des recommandations


# %%
