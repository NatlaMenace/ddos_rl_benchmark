## 🟦 **Cadrage du projet**

### Contexte
Le projet s'inscrit dans le cadre d'une étude sur la détection et la mitigation des attaques DDoS (Distributed Denial of Service) à l'aide de techniques d'apprentissage par renforcement (RL). L'objectif est d'évaluer et de comparer différentes stratégies RL pour protéger un réseau simulé contre des attaques DDoS.

### Objectifs
- Implémenter un environnement de simulation pour les attaques DDoS.
- Développer plusieurs agents RL capables de détecter et de réagir aux attaques.
- Comparer les performances des agents selon des critères définis (taux de détection, temps de réaction, impact sur le réseau).
- Documenter les résultats et proposer des pistes d'amélioration.

### Contraintes
- Utiliser Python et des bibliothèques RL standards (e.g., OpenAI Gym, Stable Baselines).
- Assurer la reproductibilité des expériences.
- Respecter un cadre éthique dans la simulation des attaques.

### Livrables
- Code source complet et documenté.
- Rapport détaillé présentant la méthodologie, les résultats et les analyses.
- Présentation orale synthétisant les points clés du projet.

### Planification
1. Recherche bibliographique et définition de l'environnement (Semaine 1-2)
2. Implémentation des agents RL (Semaine 3-5)
3. Expérimentations et collecte des données (Semaine 6-7)
4. Analyse des résultats et rédaction du rapport (Semaine 8-9)
5. Préparation de la présentation finale (Semaine 10)

## 🟦 Phase 1 — Mise en place du projet

### Création de l’environnement Python
Un environnement virtuel a été créé avec :
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Installation des dépendances
Les dépendances suivantes ont été installées :
```
pip install numpy pandas matplotlib seaborn scikit-learn
pip install gymnasium
pip install stable-baselines3
pip install kagglehub
pip install pyarrow
```

### Structure du projet
Mise en place de l’architecture standard :
src/
    agents/
    envs/
    data/
data/raw/

### Téléchargement du dataset CIC-DDoS2019
Le dataset a été téléchargé automatiquement grâce au script :
python -m src.data.download_cicddos2019

### Test de lecture
Un test dans main.py a permis de confirmer la lecture d’un fichier Parquet :
```
df = pd.read_parquet("data/raw/cicddos2019/UDP-training.parquet")
```

## 🟦 Phase 2 — Prétraitement & représentation des états RL

### 🎯 Objectifs
- Charger et fusionner les fichiers bruts du dataset CIC-DDoS2019.  
- Nettoyer, sélectionner et normaliser les features.  
- Structurer les données sous une forme exploitable pour l'apprentissage par renforcement.

### 🔧 Chargement des données brutes
Le pipeline complet de prétraitement est implémenté dans :
```
src/data/preprocessing.py
```

Le chargement fusionne automatiquement tous les fichiers `.parquet` du dossier :
```
data/raw/cicddos2019/
```

Le dataset complet contient :
- **431 371 lignes**  
- **79 colonnes**

### 🧽 Nettoyage des données
- Suppression des colonnes entièrement vides  
- Remplacement des valeurs manquantes (`NaN`) par **0**  
- Ajout d’une colonne `__source_file__` pour la traçabilité  

### 🧩 Sélection des features
Une liste de features candidates a été définie.  
Sur celles proposées, **8** étaient présentes et utilisées :

- Flow Duration  
- Tot Fwd Pkts  
- Tot Bwd Pkts  
- TotLen Fwd Pkts  
- TotLen Bwd Pkts  
- Flow Byts/s  
- Flow Pkts/s  
- Protocol  

La cible est : **Label**

### 📏 Normalisation & Split

- Standardisation via **StandardScaler()**  
- Découpage train/test : **80% / 20%**, stratifié  
- Résultats :
```
X_train : (345096, 8)
X_test  : (86275, 8)
```

### 💾 Sauvegarde des données prétraitées

Les objets suivants sont générés dans :
```
data/processed/
    X_train.npy
    X_test.npy
    y_train.npy
    y_test.npy
    scaler.pkl
```

### ▶️ Exécution du pipeline
```
python -m src.data.preprocessing
```

## 🟦 Phase 3 — Formulation RL & baseline supervisée

### 🎯 Objectifs
- Définir la formulation RL du problème de détection DDoS (MDP).
- Implémenter un environnement Gymnasium basé sur les données prétraitées.
- Mettre en place une baseline supervisée pour comparer les performances avec le RL.

### 🧠 Formulation RL (MDP)

- **États (S)** : vecteur de 8 features normalisées issu de `X_train` / `X_test`.
- **Actions (A)** :  
  - 0 = trafic normal  
  - 1 = attaque DDoS

- **Récompense (R)** :  
  - +1 si l’action correspond au label réel  
  - −2 pour un faux négatif (attaque non détectée)  
  - −1 pour un faux positif (trafic normal classé comme attaque)

- **Transitions** : l’agent parcourt des exemples du dataset, dans un ordre aléatoire à chaque épisode.

### 🧩 Environnement Gym — `DDoSDatasetEnv`

Implémenté dans :
```text
src/envs/ddos_env.py
```

### 📊 Résultats de la baseline supervisée

L'exécution de la baseline RandomForest produit automatiquement plusieurs fichiers utiles pour l’analyse :

- `reports/baseline_report.md` — Rapport Markdown complet (rapport de classification + matrice de confusion en tableau).
- `reports/confusion_matrix.png` — Visualisation graphique de la matrice de confusion.
- `data/processed/baseline_random_forest.joblib` — Modèle entraîné sauvegardé pour référence.

Commande exécutée :
```
```bash
python -m src.agents.baseline_supervised
```

Ces éléments serviront de point de comparaison lors de la Phase 6 (expérimentations RL).

## 🟦 Phase 4 — Implémentation Q-Learning (DQN)

### 🎯 Objectifs
- Implémenter une version Deep Q-Learning (DQN) adaptée aux états continus.
- Connecter l’agent DQN à l’environnement `DDoSDatasetEnv`.
- Générer un premier ensemble de courbes de récompense pour comparaison ultérieure avec PPO.

### 🧩 Agent DQN

L’agent DQN est implémenté dans :
```
src/agents/dqn_agent.py
```
Caractéristiques :
- Réseau Q approximé par un MLP (2 couches cachées, ReLU).
- Replay buffer (100 000 transitions).
- Stratégie ε-greedy avec décroissance linéaire.
- Réseau cible mis à jour périodiquement.

### ▶️ Entraînement DQN

Le script d’entraînement est :
```
main_train_dqn.py
```

Commande d’exemple :
```
```bash
python main_train_dqn.py --episodes 200 --device cpu
```

Les sorties sont sauvegardées dans :
```
models/dqn/
    dqn_cicddos.pt
    episode_rewards.npy
    losses.npy
```

### 📈 Interprétation des premiers résultats DQN

Le reward moyen passe d’environ **-760** au début de l’entraînement à environ **-560** sur les épisodes les plus récents.  
Cette amélioration montre que l’agent apprend progressivement à réduire ses erreurs de classification, même si un plateau apparaît après une centaine d’épisodes.  
Ce comportement est cohérent avec :

- une fonction de récompense fortement négative (FN = -2, FP = -1),  
- un dataset très volumineux (430k flux),  
- un environnement non-Markovien (chaque flux est indépendant),  
- une phase d'exploration ε-greedy encore élevée au début.

Ces résultats constituent la baseline RL initiale et seront comparés aux performances obtenues par PPO en Phase 6.

### 📁 Sorties générées par le DQN

Les fichiers produits par l'entraînement DQN sont :

```
models/dqn/
    dqn_cicddos.pt
    episode_rewards.npy
    losses.npy
```

Ils seront utilisés lors de l’analyse comparative finale (Phase 6).

## 🟦 Phase 5 — PPO (Policy Gradient)

### 🎯 Objectifs
- Implémenter un agent PPO basé sur les policy gradients.
- Réutiliser le même environnement `DDoSDatasetEnv` que pour DQN pour permettre une comparaison directe.
- Générer des métriques de classification et des courbes d’apprentissage comparables à celles de DQN.

### 🧠 Rappel théorique sur PPO

PPO (Proximal Policy Optimization) est un algorithme d’apprentissage par renforcement basé sur les policy gradients.  
L’idée principale est de mettre à jour les paramètres de la politique \(\pi_\theta(a \mid s)\) en maximisant un objectif de type gradient de politique, tout en **limitant la taille des mises à jour** pour éviter les instabilités.

L’objectif PPO utilise un ratio entre la nouvelle et l’ancienne politique :

\[
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}
\]

et maximise une fonction **“clippée”** :

\[
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right) \right]
\]

où \(A_t\) est l’avantage (estimé typiquement via GAE) et \(\epsilon\) un hyperparamètre de clip (ex. 0.2).  
Cette forme “clippée” empêche les mises à jour trop agressives qui dégraderaient brutalement la politique.

Dans ce projet :
- la politique est un MLP (MlpPolicy, Stable-Baselines3),  
- l’algorithme utilise PPO avec GAE, objectif clippé, et optimisation par mini-batchs.

### ⚙️ Implémentation PPO

L’agent PPO est implémenté avec **Stable-Baselines3** :

- Fichier de configuration / helper :
  - `src/agents/ppo_agent.py` (classe `PPOConfig` et fonction `make_ppo_model(...)`).
- Script d’entraînement :
  - `main_train_ppo.py`

L’entraînement est lancé avec une commande du type :
```bash
python main_train_ppo.py --total-timesteps 500000 --device cpu --max-steps 1000
```

Caractéristiques :
- environnement : `DDoSDatasetEnv(split="train")` pour l’entraînement, `split="test"` pour l’évaluation,
- politique : MlpPolicy (SB3),
- hyperparamètres (exemple) : learning_rate = 3e-4, gamma = 0.99, clip_range = 0.2, n_steps = 2048, n_epochs = 10.

### 📊 Résultats expérimentaux PPO (première passe)

Évaluation sur le set de test (10 000 échantillons, labels multi-classes) :

- Accuracy globale : **22,40 %**
- Macro F1 : **0,023**
- Weighted F1 : **0,083**

Le rapport de classification montre que :
- la classe `0` (majoritaire) atteint une précision ≈ 0,23 et un recall ≈ 0,999,
- toutes les autres classes (1 à 15) ont une précision, un recall et un F1-score de **0,0**.

La matrice de confusion indique que :
- presque toutes les prédictions sont faites en classe `0`,
- les classes minoritaires ne sont quasiment jamais prédites.

Interprétation :
- PPO, dans cette configuration de récompense et avec des labels multi-classes fortement déséquilibrés, **collabe vers une politique triviale** : prédire systématiquement (ou quasi systématiquement) la classe majoritaire.
- L’accuracy ~22 % correspond à peu près à la proportion de la classe `0` dans le test, ce qui montre que l’agent n’exploite pas l’information fine des autres classes.
- Pour la détection DDoS multi-attaques, ce comportement est insuffisant : le modèle ne détecte pas les types d’attaque spécifiques (classes 1 à 15).

Ces résultats motivent :
- soit une reformulation du problème en **binaire** (benign vs attaque),
- soit un rééquilibrage de la récompense (pondération plus forte des classes d’attaque),
- soit une combinaison RL + supervision pour améliorer la sensibilité aux classes minoritaires.

### 📁 Sorties générées par PPO

L’entraînement PPO produit les fichiers suivants :

```
models/ppo/
    ppo_cicddos.zip        # modèle PPO sauvegardé
    episode_rewards.npy    # rewards par épisode (évaluation)
reports/
    ppo_report.md          # rapport de classification (test)
    ppo_confusion_matrix.png  # matrice de confusion PPO
```
