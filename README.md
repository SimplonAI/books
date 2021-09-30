# BYAM - Yet Another Model for Books recommendation

Byam est une API en apprentissage continue qui permet de recommander des livres à des utilisateurs, quelque soit leur ancienneté, via différent modèles de recommandations (Popularité, Contenu et Collaboratif).

Le principe est d'apprendre de l'utilisateur dès son inscription en lui proposant de noter une dizaine de livres. La liste de livres recommandés à noter s'améliorant à chaque notation.

Ainsi trois types de listes de livres à noter font leur apparition durant le processus d'inscription:
1. Popularity-based : On ne connaît encore rien de l'utilisateur, on lui affiche les livres les plus populaires du moment en espérant qu'il en ait lu au moins un.
2. Content-based : A chaque livre noté, on rafraichit la liste de livres proposés à noter par des livres similaires à ceux qu'il vient de noter. Ainsi, il y a plus de chances qu'il ait déjà lu ces livres
3. Collaborif : Au bout de 5 livres notés, on commence à enrichir la liste avec des livres provenant de notre algorithme collaboratif. 

## Prérequis
* Python 3.9
* PostgreSQL

## Configuration
1. Pour commencer, il va falloir cloner le répository et créer un environnement virtuel : 
```bash
git clone https://github.com/SimplonAI/books
cd books
python -m venv venv
```
2. Activer ensuite votre environnement virtuel python :
* Sous Linux
```bash
source venv/bin/activate
```
* Sous Windows
```bash
./venv/Scripts/activate
```
3. Il faut ensuite installer les dépendances :
```bash
pip install -r requirements.txt
```
4. Copier et renommer le fichier config_example.yml et y ajouter vos identifiants de connexion à la base de données :
```bash
cp config_example.yml config.yml
```

## Installation
Avant de pouvoir utiliser l'API, il va falloir créer la base de données puis insérer les données des fichiers CSV :
```bash
python create_db.py
# Choisir l'option 1
python insert_db.py
```
En cas d'erreur de connexion à la BDD, veuillez vérifier que vous avez bien configuré votre config.yml.

Par la suite, vous pouvez entraîner les modèles de recommendations :
```bash
# Choisir l'option 1
python main.py
```

## Lancement
Vous disposez de trois routes (fonctionnalités) différentes correspondant aux 3 étapes de recommandations mentionnées ci-dessus.

La première fonctionnalité (option 2) est la recommandation basé sur la popularité.

La deuxième fonctionnalité (option 3) est la recommandation basé sur le contenu. Il vous sera demandé comme argument GET (ou ici via un prompt) le titre du livre (exemple: The Nix).

La troisième fonctionnalité (option 4) est la recommandation collaborative. Il vous sera demandé comme argument GET (ou ici via un prompt) l'id de l'utilisateur pour lequel faire la prédiction (exemple: 43675)

Enfin il sera possible de lancer un serveur de développement en tapant directement :
```bash
python main.py -r
```