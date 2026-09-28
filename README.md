# Cycling Webapp

Application web développée avec [Streamlit](https://streamlit.io/) pour explorer et analyser des données cyclistes.

Le projet permet notamment de charger les données de courses, filtrer les résultats, calculer des statistiques et produire différents classements et agrégations.

L'objectif du projet est de construire un classement historique du cyclisme féminin, rassemblant autant que possible les différentes époques, et de permettre l'exploitation de ces données par différents moyens d'analyse.

Les données compilées ne sont pas exhaustives et regroupent uniquement les données nécessaires au classement. L'intégralité des courses et des résultats n'est donc pas disponible. Par ailleurs, la catégorisation des courses utilisée dans le projet ne représente pas nécessairement leur catégorie historique.

## Fonctionnement

Les données sont stockées dans des fichiers CSV dans `data/raw/`. Elles sont chargées et enrichies par le module `dataloader`, puis utilisées par l'application Streamlit.

La chaîne de traitement principale est organisée autour de :

- `load.py` : chargement et enrichissement des données ;
- `validate.py` : vérification de l'intégrité des données brutes et enrichies ;
- `filters.py` : filtrage des données selon différents critères ;
- `agregations.py` : agrégations et synthèses ;
- `ranking.py` : calcul des classements et attribution des points ;
- `stats.py` : calcul des statistiques.

L'affiche dans la webapp est organisée autour de :

- `home.py` : page d'accueil ;
- `resultats.py` : page de consultation des résultats et palmarès ;
- `classement.py` : page de consultation du classement complet ;
- `cycliste.py` : page de consultation des données et statistiques par cycliste ;
- `nation.py` : page de consultation des données et statistiques par nation ;
- `course.py` : page de consultation des données et statistiques par course ;  
- `annee.py` : page de consultation des données et statistiques par année.

## Structure du projet

```text
cycling-webapp/
├── app.py
├── LICENSE
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
│
├── data/
│   └── raw/
│       ├── countries.csv
│       ├── riders.csv
│       ├── races.csv
│       ├── editions.csv
│       └── results.csv
│
├── dataloader/
│   ├── __init__.py
│   ├── load.py
│   └── validate.py
│
├── pages/ 
│   ├── __init__.py
│   ├── annee.py
│   ├── classement.py
│   ├── course.py
│   ├── cycliste.py
│   ├── home.py
│   ├── nation.py
│   └── resultats.py
│
└── src/
    ├── __init__.py
    ├── filters.py
    ├── agregations.py
    ├── ranking.py
    └── stats.py
```

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) pour gérer l'environnement Python et les dépendances.

Après avoir cloné le dépôt :

```bash
uv sync
```

Cette commande crée ou utilise l'environnement virtuel du projet et installe les dépendances définies dans `pyproject.toml` et verrouillées dans `uv.lock`.

## Lancer l'application

Pour démarrer l'application en local :

```bash
uv run streamlit run app.py
```

Il est également possible d'utiliser directement l'environnement virtuel :

```bash
streamlit run app.py
```

L'application est alors accessible depuis le navigateur à l'adresse indiquée par Streamlit, généralement :

```text
http://localhost:8501
```

## Données

Les données sources sont conservées dans `data/raw/` sous forme de fichiers CSV.

Les fichiers sont volontairement séparés des traitements afin de pouvoir mettre à jour régulièrement les données sans modifier le code de l'application.

Une mise à jour des données consiste donc principalement à remplacer ou modifier les fichiers CSV concernés, puis à versionner ces changements avec Git.

## Sources des données

Les résultats et informations utilisés par l'application sont issus de données
publiées officiellement par les organisateurs des épreuves, qui ont été compilées par des sources publiques spécialisées.

**Sources principales :**
- [Cyclebase](https://www.cyclebase.nl)
- [First Cycling](https://firstcycling.com)
- [Pro Cycling Stats](https://www.procyclingstats.com)
- [CQ Ranking](https://cqranking.com/women/asp/gen/start.asp)
- [Wikipedia](https://fr.wikipedia.org/wiki/Wikipédia:Accueil_principal)

Ces références sont fournies à titre documentaire et permettent d'identifier
les sources utilisées pour constituer les fichiers de données du projet.

## Développement

Pour modifier le projet :

1. créer ou mettre à jour l'environnement avec `uv` ;
2. développer et tester localement avec Streamlit ;
3. vérifier que l'application fonctionne avec les données présentes dans `data/raw/` ;
4. enregistrer les modifications avec Git ;
5. pousser les changements vers le dépôt distant.

## Licence

Le code de ce projet est distribué sous licence MIT.
Voir le fichier [LICENSE](LICENSE).

Les données utilisées par l'application proviennent de sources publiques
et restent soumises, le cas échéant, aux conditions applicables à leurs
sources respectives.
