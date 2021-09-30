from fastapi import FastAPI, Response, status
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

content_model = ContentBasedModel(None)
collab_model = CollaborativeBasedModel()

@app.get("/popularity")
async def popularity_rec():
    _, books = await DataManager(db, keep_data=True).data
    return PopularityBasedModel(books).predict().to_dict('records')

@app.get("/content-based")
async def content_based(response: Response, book_title: str = ""):
    if book_title == "":
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "book_title doit être une chaîne de caractères non vide"}
    if content_model.books is None:
        _, books = await DataManager(db, keep_data=True).data
        content_model.books = books
    content_model.load()
    return content_model.predict(book_title).to_dict('records')

@app.get("/collaborative")
async def collab_based(response: Response, user_id: int = 0):
    if user_id <= 0:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "user_id doit être un entier positif différent de 0"}
    collab_model.load()
    _, book_ids = collab_model.predict(user_id)
    _, books = await DataManager(db, keep_data=True).data
    book_titles = [ title.decode("utf-8") for title in book_ids[0, :10].numpy().tolist()]
    return books[books["book_title"].isin(book_titles)].to_dict('records')
