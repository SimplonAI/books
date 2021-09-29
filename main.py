from __future__ import annotations
import argparse
import asyncio
import os
import sys
from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem
import tensorflow as tf
from tqdm import tqdm
from ml.data_manager import DataManager
from ml.content_model import ContentBasedModel
from ml.collab_model import CollaborativeBasedModel
from ml.pop_model import PopularityBasedModel

class Main:
    """Classe de lancement du programme principal."""

    def __init__(self):
        # Titre de l'application
        self.title = "BYAM - Yet Another Model for Books"
        # Slogan de l'application
        self.slogan = "L'API qui recommande les livres pour vos utilisateurs !"
        if not os.path.isfile(os.path.join("config.yml")):
            sys.exit("Vous devez configurer votre application avec un config.yml pour qu'elle puisse s'exécuter !")
        # Construction du menu
        self.__menu = self._build_menu()
        # Construction des arguments en ligne de commande
        self.__parser = self._build_parser()
        from db import db
        print("Chargement des données...")
        data_manager = DataManager(db, keep_data=True)
        loop = asyncio.new_event_loop()
        # Chargement des données de la bdd
        _, books = loop.run_until_complete(data_manager.data)
        self.content_model = ContentBasedModel(books)
        self.collab_model = CollaborativeBasedModel()

    def _build_menu(self) -> ConsoleMenu:
        """Crée un menu console avec les différentes fonctionnalités pouvant être exécutées

        Returns:
            ConsoleMenu: Un objet ConsoleMenu
        """
        menu = ConsoleMenu(
            self.title,
            self.slogan,
            prologue_text="Choix de la fonctionnalité à exécuter (chiffre correspondant)",
        )
        menu.append_item(FunctionItem("Entrainement des modèles", self._train))
        menu.append_item(
            FunctionItem("Recommandation de popularité", self._popularity_rec)
        )
        menu.append_item(
            FunctionItem("Recommandation basé sur le contenu", self._content_rec)
        )
        menu.append_item(
            FunctionItem(
                "Recommandation collaborative pour un utilisateur", self._collab_rec
            )
        )
        menu.append_item(
            FunctionItem("Lancer un serveur de développement pour l'API", self._run_api)
        )
        return menu

    def _build_parser(self) -> argparse.ArgumentParser:
        """Défini les différents arguments que le programme peut prendre en compte de

        Returns:
            argparse.ArgumentParser: Un object ArgumentParser
        """
        parser = argparse.ArgumentParser(prog=self.title, description=self.slogan)
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "-t", "--train", help="Entrainement des modèles", action="store_true"
        )
        group.add_argument(
            "-p",
            "--poprec",
            help="Recommandation de popularité",
            action="store_true",
        )
        group.add_argument(
            "-c",
            "--contentrec",
            help="Recommandation basé sur le contenu",
            action="store_true",
        )
        group.add_argument(
            "-b",
            "--collabrec",
            help="Recommandation collaborative pour un utilisateur",
            action="store_true",
        )
        group.add_argument(
            "-r",
            "--runapi",
            help="Lancer un serveur de développement pour l'API",
            action="store_true",
        )
        parser.add_argument("--path", help="Path to save the model", type=str)
        return parser

    def run(self):
        if len(sys.argv) == 1:
            self.__menu.show()
        else:
            args = self.__parser.parse_args()
            if args.train:
                self._train(args.path)
            elif args.poprec:
                self._popularity_rec()
            elif args.contentrec:
                self._content_rec()
            elif args.collabrec:
                self._collab_rec()
            elif args.runapi:
                self._run_api()

    def _train(self, path: (str | None) = None):
        """Fonction de lancement de l'entraînement des modèles

        Args:
            path (str): Chemin où l'on veut enregistrer les modèles entraînés
        """
        from db import db
        loop = asyncio.new_event_loop()
        data_manager = DataManager(db, keep_data=True)
        data, books = loop.run_until_complete(data_manager.data)
        train_models = [lambda: ContentBasedModel(books).train_save(), lambda: CollaborativeBasedModel().train_save(data, books)]
        for train_model in tqdm(train_models, desc="Entrainement des modèles"):
            train_model()
        input("Entrée pour continuer...")

    def _popularity_rec(self):
        from db import db
        print("Chargement des données...")
        data_manager = DataManager(db, keep_data=True)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(data_manager._load(db))
        # Chargement des données de la bdd
        _, books = loop.run_until_complete(data_manager.data)
        pop_model = PopularityBasedModel(books)
        print(pop_model.predict())
        input("Entrée pour continuer...")

    def _content_rec(self):
        book_title = input("Titre du livre : ")
        self.content_model.load()
        print(self.content_model.predict(book_title))
        input("Entrée pour continuer...")


    def _collab_rec(self):
        try:
            user_id = int(input("ID de l'utilisateur : "))
            self.collab_model.load()
            print(self.collab_model.predict(user_id))
        except:
            print("Vous devez entrer l'id d'un utilisateur (exemple 43675) !")
        input("Entrée pour continuer...")
    
    def _run_api(self):
        print("Pas encore implémenté !")
        input("Entrée pour continuer...")
        


if __name__ == "__main__":
    tf.get_logger().setLevel(3)
    tf.autograph.set_verbosity(1)
    main = Main()
    main.run()
