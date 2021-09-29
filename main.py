from __future__ import annotations
import argparse
from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem
import sys


class Main:
    """Classe de lancement du programme principal.
    """
    def __init__(self):
        # Titre de l'application
        self.title = "BookPi"
        # Slogan de l'application
        self.slogan = "L'API qui recommande les livres pour vos utilisateurs !"
        # Construction du menu
        self.__menu = self._build_menu()
        # Construction des arguments en ligne de commande
        self.__parser = self._build_parser()

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
            elif args.contentrec:
                self._content_rec()
            elif args.collabrec:
                self._collab_rec()
            elif args.runapi:
                self._run_api()

    def _train(self, path: (str | None)=None):
        """Fonction de lancement de l'entraînement des modèles

        Args:
            path (str): Chemin où l'on veut enregistrer les modèles entraînés
        """
        pass

    def _content_rec(self):
        pass

    def _collab_rec(self):
        pass

    def _run_api(self):
        pass


if __name__ == "__main__":
    main = Main()
    main.run()
