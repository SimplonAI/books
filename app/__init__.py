from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from ml.pop_model import PopularityBasedModel
from ml.data_manager import DataManager
from db import db
from ml.content_model import ContentBasedModel
from ml.collab_model import CollaborativeBasedModel

description = """
L'API qui recommande les livres pour vos utilisateurs ! 🚀


## Recommandations
* Basé sur la popularité
* Basé sur le contenu
* Collaborative
"""

app = FastAPI(
    title="BYAM - Yet Another Model for Books",
    description=description,
    version="0.1.0",
    contact={"name": "Byam support", "email": "support@byam.fr"},
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

class BookResponse(BaseModel):
    book_id: int
    book_title: str
    book_original_title: str
    book_authors: str
    book_language_code: str
    book_original_publication_year: int
    book_average_rating: float
    book_isbn: str
    book_isbn13: str
    book_ratings_count: int
    book_work_ratings_count: int
    book_work_text_reviews_count: int
    book_ratings_1: int
    book_ratings_2: int
    book_ratings_3: int
    book_ratings_4: int
    book_ratings_5: int
    book_image_url: str
    book_small_image_url: str
    book_books_count: int
    book_work_id: int
    book_best_book_id: int
    book_goodreads_book_id: int
    goodreads_book_id: int
    tags: str

content_model = ContentBasedModel(None)
collab_model = CollaborativeBasedModel()

@app.get("/popularity", response_model=list[BookResponse])
async def popularity_rec():
    _, books = await DataManager(db, keep_data=True).data
    return PopularityBasedModel(books).predict().to_dict('records')

@app.get("/content-based", response_model=list[BookResponse])
async def content_based(response: Response, book_title: str = ""):
    if book_title == "":
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "book_title doit être une chaîne de caractères non vide"}
    if content_model.books is None:
        _, books = await DataManager(db, keep_data=True).data
        content_model.books = books
    content_model.load()
    return content_model.predict(book_title).to_dict('records')

@app.get("/collaborative", response_model=list[BookResponse])
async def collab_based(response: Response, user_id: int = 0):
    if user_id <= 0:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "user_id doit être un entier positif différent de 0"}
    collab_model.load()
    _, book_ids = collab_model.predict(user_id)
    _, books = await DataManager(db, keep_data=True).data
    book_titles = [ title.decode("utf-8") for title in book_ids[0, :10].numpy().tolist()]
    return books[books["book_title"].isin(book_titles)].to_dict('records')

@app.on_event("startup")
def startup_event():
    print("Vous pouvez accéder à la documentation de l'api en allant sur :")
    print("http://127.0.0.1:8000/docs")