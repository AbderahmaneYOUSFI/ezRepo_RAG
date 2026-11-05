# ez-repo

**Posez des questions sur le code d'un projet GitHub, et obtenez des réponses qui citent les fichiers.**

Exemple : « Où est définie la fonction `add_one` ? » → réponse + lien vers le fichier et les lignes concernés.

ez-repo est **100 % local et gratuit** : tout tourne sur votre PC, sans clé d'API ni service payant.

> 🔒 ez-repo **n'exécute jamais** le code des projets analysés. Il le lit seulement comme du texte.

---

## Ce que fait l'application

1. Vous collez l'URL d'un repository GitHub public écrit en Python.
2. ez-repo télécharge le code et repère les classes, les fonctions, les imports et les appels.
3. Il range ces informations dans un **graphe** : quelle fonction appelle quelle autre, quelle classe hérite de quelle autre, quel fichier importe quel fichier.
4. Vous posez une question en langage naturel.
5. ez-repo retrouve les morceaux de code utiles, puis un modèle d'IA rédige la réponse en citant les fichiers et les lignes.

## Comment ça marche

Le projet utilise la technique **GraphRAG**.

- **RAG** (*Retrieval-Augmented Generation*) : avant de répondre, on *retrouve* les bons morceaux de code et on les donne au modèle d'IA. Il répond à partir de ce code, au lieu d'inventer.
- **Graph** : en plus de la recherche par sens, on suit les liens du graphe. Si la question porte sur une fonction, on récupère aussi les fonctions qu'elle appelle.

```
  URL GitHub
      │
      ▼
  1. Téléchargement   git clone, puis liste des fichiers .py
      │
      ▼
  2. Analyse          Tree-sitter trouve classes, fonctions, imports, appels
      │
      ▼
  3. Stockage         Neo4j (graphe)  +  embeddings (sens du code)
      │
      │      Question
      │         │
      ▼         ▼
  4. Recherche        noms exacts  +  sens proche  +  voisins dans le graphe
      │
      ▼
  5. Réponse          Ollama (IA locale) → réponse + sources
```

## Les technologies et leur rôle

| Technologie | C'est quoi ? | À quoi elle sert ici |
|---|---|---|
| **Python 3.11** | Le langage du projet. | Tout le code est en Python. |
| **Git** | L'outil de gestion de versions. | Télécharger le repository (seulement la dernière version, pour aller vite). |
| **Tree-sitter** | Un *parseur* : il transforme du code en arbre (fonction → paramètres → corps…). | Trouver les classes, fonctions, imports et appels **sans exécuter** le code. |
| **Neo4j** | Une base de données *graphe* : des nœuds reliés par des relations. | Stocker les fichiers et fonctions comme nœuds, et leurs liens (`CALLS`, `IMPORTS`, `INHERITS`…). Elle sert aussi d'index de recherche par sens. |
| **Docker** | Lance un logiciel dans un « conteneur », sans l'installer sur Windows. | Démarrer Neo4j avec une seule commande. |
| **Embeddings** (`all-MiniLM-L6-v2`) | Un petit modèle qui transforme un texte en liste de 384 nombres. Deux textes au sens proche donnent des listes proches. | Chaque fonction et chaque question devient un vecteur, ce qui permet de chercher **par sens** et pas seulement par mot exact. |
| **Ollama** + `qwen2.5-coder:3b` | Ollama fait tourner des modèles d'IA (LLM) sur votre PC. Qwen Coder est un petit modèle spécialisé dans le code. | Rédiger la réponse finale à partir du code retrouvé. |
| **LangChain** | Une bibliothèque pour relier les modèles d'IA aux programmes. | Construire le *prompt* (consignes + code trouvé + question) et l'envoyer à Ollama. |
| **FastAPI** | Un framework pour créer une API web en Python. | Le **backend** : il indexe les repositories en arrière-plan et répond aux questions. |
| **Streamlit** | Crée une interface web en Python, sans HTML ni JavaScript. | Le **frontend** : la page avec la barre latérale et le chat. Il ne fait qu'appeler l'API. |
| **Pytest** / **Ruff** | Outils de tests / de vérification du style. | Vérifier que le code fonctionne et reste propre. |

## Installation (Windows)

### Logiciels à installer

- **Python 3.11 ou plus** : <https://www.python.org/downloads/> (cochez « Add to PATH »)
- **Git** : <https://git-scm.com/download/win>
- **Ollama** : <https://ollama.com/download/windows>
- **Docker Desktop** : <https://www.docker.com/products/docker-desktop/>

### Préparer le projet

Dans PowerShell, à la racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

Le fichier `.env` contient les réglages, dont le mot de passe Neo4j (`NEO4J_PASSWORD`). Changez-le **maintenant** si vous le souhaitez, avant de démarrer Neo4j la première fois.

> Si PowerShell refuse d'activer le venv : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Télécharger le modèle d'IA

```powershell
ollama pull qwen2.5-coder:3b
```

Le modèle d'embeddings se télécharge tout seul à la première utilisation (environ 90 Mo).

## Lancer l'application

Il faut 4 éléments : Neo4j, Ollama, l'API et l'interface.

```powershell
# 1. Neo4j (une seule fois : Docker le redémarre ensuite tout seul)
docker compose up -d

# 2. Ollama : il tourne normalement déjà en arrière-plan après l'installation.
#    Sinon : ollama serve

# 3. L'API, dans un premier terminal
.\.venv\Scripts\Activate.ps1
ez-repo-api

# 4. L'interface, dans un deuxième terminal
.\.venv\Scripts\Activate.ps1
streamlit run src/ez_repo/ui.py
```

Ouvrez ensuite <http://localhost:8501>.

| Adresse | Ce que c'est |
|---|---|
| <http://localhost:8501> | L'application |
| <http://127.0.0.1:8000/docs> | La documentation interactive de l'API |
| <http://localhost:7474> | L'interface de Neo4j, pour voir le graphe |

> `http://127.0.0.1:8000` seul affiche `{"detail": "Not Found"}` : c'est normal, l'API n'a pas de page d'accueil.

## Utiliser l'application

1. Dans la barre latérale, collez une URL, par exemple `https://github.com/pypa/sampleproject`.
2. Cliquez sur **Analyze repository** et attendez la fin de l'indexation.
3. Sélectionnez le repository et posez votre question dans le chat.
4. Les sources s'affichent sous la réponse, avec un lien vers GitHub.

Sur un PC sans carte graphique, une réponse prend entre 10 et 90 secondes.

## Problèmes fréquents

**« Neo4j rejected the credentials »**

Le mot de passe de `.env` ne correspond pas à celui de Neo4j. Neo4j retient le mot de passe de sa **toute première** création et ignore les changements ultérieurs de `.env`. Pour repartir de zéro (les repositories indexés seront à réimporter) :

```powershell
docker compose down
Remove-Item -Recurse -Force data\neo4j
docker compose up -d
```

Redémarrez ensuite l'API, car elle garde sa connexion à Neo4j.

**« error while attempting to bind on address ('127.0.0.1', 8000) »**

Une ancienne API tourne encore. Arrêtez-la, puis relancez :

```powershell
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess
ez-repo-api
```

**« Cannot reach the ez-repo API »** dans l'interface

L'API n'est pas lancée : démarrez `ez-repo-api` dans un autre terminal.

## Tests

```powershell
pytest                 # tests unitaires, aucun service nécessaire
pytest -m integration  # vérifie Neo4j et Ollama (ils doivent être démarrés)
ruff check .           # vérification du style
```

## Organisation du code

```
ez-repo/
├── src/ez_repo/
│   ├── ingest.py       # 1. télécharge le repository et liste les fichiers .py
│   ├── parser.py       # 2. Tree-sitter : classes, fonctions, imports, appels
│   ├── embeddings.py   #    transforme le code en vecteurs
│   ├── graph.py        # 3. écrit le graphe et les vecteurs dans Neo4j
│   ├── retriever.py    # 4. recherche le code utile pour une question
│   ├── qa.py           # 5. construit le prompt et interroge Ollama
│   ├── tasks.py        #    indexation en arrière-plan
│   ├── api.py          #    backend FastAPI
│   ├── ui.py           #    interface Streamlit
│   └── config.py       #    lit les réglages du fichier .env
├── tests/              # tests Pytest
├── docs/               # documentation détaillée
├── docker-compose.yml  # configuration de Neo4j
└── .env.example        # modèle du fichier de réglages
```

Pour aller plus loin (modèle du graphe, routes de l'API, limites), voir [docs/details-techniques.md](docs/details-techniques.md).

## Auteur

**Abderahmane Yousfi**

## Licence

MIT
