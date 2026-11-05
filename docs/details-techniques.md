# ez-repo : détails techniques

Ce document complète le [README](../README.md). Il décrit le fonctionnement interne :
modèle du graphe, recherche, ingestion et API.

## Modèle de graphe (Neo4j)

```
(:Repository {id})-[:CONTAINS]->(:File)
(:File)-[:DEFINES]->(:Symbol:Class | :Symbol:Function)     définitions de premier niveau
(:Symbol)-[:DEFINES]->(:Symbol)                            méthodes, définitions imbriquées
(:File)-[:IMPORTS]->(:File | :ExternalModule)
(:Function)-[:CALLS]->(:Function | :Class)
(:Class)-[:INHERITS]->(:Class)
```

- Fonctions et méthodes portent le label `Function` (propriété `kind` : `function` / `method`).
- Chaque nœud porte `repo_id` (`owner/repo` en minuscules) et l'identifiant stable du parser
  (`chemin`, `chemin::Nom.qualifié`, `module:nom`) ; `(repo_id, id)` est unique par label,
  donc deux repositories ne partagent jamais un nœud.
- Les symboles portent un `embedding` (384 dim.) interrogé via l'index vectoriel
  `symbol_embeddings` (cosinus).
- Réindexation idempotente : tout est `MERGE`, horodaté par un identifiant de run ; ce que
  le run n'a pas écrit est supprimé (par lots). Un symbole dont le texte indexé n'a pas
  changé garde son embedding (pas de recalcul).

```python
from ez_repo.graph import GraphStore, index_repository, repo_id_for
from ez_repo.ingest import ingest_repository
from ez_repo.parser import parse_repository

result = ingest_repository("https://github.com/pypa/sampleproject")
with GraphStore() as store:
    stats = index_repository(store, repo_id_for(result.repo), parse_repository(result.files),
                             url=result.repo.url)
```

Le modèle d'embeddings tourne sur CPU (~30 symboles/s) et n'est chargé qu'une fois par
processus. Le texte indexé d'un symbole contient le nom qualifié, le type, le fichier,
la signature, les décorateurs, la docstring et un extrait du code.

`CALLS` et `INHERITS` ne sont créés que lorsque la cible est résolue sans ambiguïté
(définition du même fichier, fonction englobante, `from x import nom` interne au repo,
`self.methode()` définie dans la classe). Les appels d'attributs, les builtins et les
noms masqués ne produisent aucune relation.

## Questions (GraphRAG)

```python
from ez_repo.graph import GraphStore
from ez_repo.qa import ask_repository

with GraphStore() as store:
    answer = ask_repository(store, "pypa/sampleproject", "Où se trouve la fonction add_one ?")
print(answer.answer)
for source in answer.cited_sources:
    print(source.label, source.name, source.location)   # ex. S1 pkg.core.greet pkg/core.py:12-17
```

1. **Recherche exacte** : noms écrits dans la question (`greet`, `Greeter.run`, `pkg.core`,
   `pkg/core.py`), sensibles à la casse (« app » n'est pas la classe `App`).
2. **Recherche vectorielle** dans l'index `symbol_embeddings`, filtrée par `repo_id`
   (scores sous `RETRIEVAL_MIN_SCORE` ignorés).
3. **Graphe** : voisins `CALLS` / `INHERITS` / `DEFINES` des meilleurs symboles et `IMPORTS`
   des fichiers cités, sur `RETRIEVAL_DEPTH` niveaux (1 ou 2).

Les symboles sont dédoublonnés, classés, limités à `RETRIEVAL_MAX_SYMBOLS` et rendus avec
leurs numéros de ligne sous `QA_MAX_CONTEXT_CHARS` caractères. Sans contexte pertinent, le
LLM n'est pas appelé et la réponse indique que l'information est insuffisante.

Le modèle cite des étiquettes (`[S1]`) ; les sources renvoyées viennent des métadonnées
Neo4j, jamais du texte généré. Les étiquettes inexistantes sont retirées
(`invalid_citations`), les chemins `.py` inconnus du repository sont signalés
(`unknown_paths`), une réponse coupée par `OLLAMA_NUM_PREDICT` est marquée `incomplete`.

Sur CPU, une réponse prend ~10 à 90 s selon la longueur (`OLLAMA_TIMEOUT`, 180 s par défaut).

## Ingestion

- URL acceptée : `https://github.com/<owner>/<repo>` (variantes `.git`, `/tree/...`, sans schéma).
- Clone superficiel (`--depth 1`), sans sous-modules, sans LFS, sans invite d'identifiants.
- Seuls les fichiers `.py` UTF-8 sont lus ; `.git`, `.venv`, `node_modules`, `__pycache__`,
  `build`, `dist`… sont ignorés. Limites configurables : `MAX_FILE_SIZE`, `MAX_FILES`,
  `MAX_TOTAL_SIZE`, `CLONE_TIMEOUT`.

## API

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | l'API répond (`{"status": "ok", "version": ...}`) |
| GET | `/health/services` | Neo4j joignable, modèle Ollama installé |
| GET | `/repositories` | repositories indexés (statut, compteurs, `commit_sha`, tâche en cours) |
| POST | `/repositories` | `{"url": "https://github.com/owner/repo"}` → **202** + tâche d'indexation |
| GET | `/tasks`, `/tasks/{id}` | état : `pending`, `running`, `completed`, `failed` (+ `stage`, `error`, `result`) |
| POST | `/chat` | `{"repo_id": "owner/repo", "question": "..."}` → réponse, sources, `incomplete`... |
| DELETE | `/repositories/{owner}/{repo}` | supprime le graphe, les embeddings et le clone local |

```powershell
$task = Invoke-RestMethod -Method Post http://127.0.0.1:8000/repositories `
  -ContentType application/json -Body '{"url": "https://github.com/pypa/sampleproject"}'
Invoke-RestMethod http://127.0.0.1:8000/tasks/$($task.id)      # jusqu'à "completed"
Invoke-RestMethod -Method Post http://127.0.0.1:8000/chat -ContentType application/json `
  -Body '{"repo_id": "pypa/sampleproject", "question": "Where is add_one defined?"}'
```

Codes d'erreur : 422 (URL ou données invalides), 404 (repository ou tâche inconnus),
409 (indexation en cours, ou dernière indexation échouée), 429 (file d'indexation pleine,
ou une autre question en cours), 503 (Neo4j / Ollama / modèle indisponible),
504 (Ollama trop lent, `OLLAMA_TIMEOUT`).

### Indexation en tâche de fond

L'indexation tourne dans un thread du processus API :
une seule à la fois (`INDEX_MAX_CONCURRENT`), au plus `INDEX_MAX_PENDING` en attente,
jamais deux fois le même repository en parallèle. Les tâches sont **gardées en mémoire** :
elles disparaissent au redémarrage de l'API (un repository resté « en cours » apparaît
alors `interrupted` : réimportez-le). Lancez donc l'API en **un seul processus**.

### Liens GitHub

Le SHA du commit cloné est lu après le clone (`git rev-parse`) et
enregistré à la fin de l'indexation. Un lien n'est produit que si le SHA est valide et
que l'URL enregistrée correspond au repository ; sinon la source est affichée sans lien
(cas des repositories indexés avant cette version : réimportez-les pour avoir les liens).
Les liens écrits par le modèle lui-même ne sont jamais rendus cliquables.

### Ressources

Un seul driver Neo4j par processus, modèle d'embeddings et client Ollama
chargés une fois, une seule génération Ollama à la fois. L'interface Streamlit ne charge
aucun modèle : elle ne fait qu'appeler l'API.

## Ressources machine

Pensé pour un PC 16 Go sans GPU dédié : Neo4j est limité à ~1,5 Go,
`qwen2.5-coder:3b` utilise ~2-3 Go, les embeddings tournent sur CPU.
