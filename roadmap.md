# Roadmap – Projet RL pour la détection d’attaques DDoS (CIC-DDoS2019)

Ce fichier sert de **trace écrite globale** et de **plan de travail** pour le projet :

- Description du jeu de données CIC-DDoS2019
- Hypothèse théorique : PPO vs Q-Learning
- Axes de comparaison des modèles
- Plan du rapport scientifique
- Roadmap de développement Python (implémentation)

---

## 1. Contexte et objectifs du projet

Dans le cadre de la matière « Fondements de l’intelligence artificielle », nous étudions la **détection d’attaques DDoS** à partir du jeu de données **CIC-DDoS2019**.  
L’objectif n’est pas uniquement de construire un modèle de classification, mais de :

1. **Formuler le problème comme une tâche d’apprentissage par renforcement (RL)**, où un agent prend des décisions de détection au fil du temps à partir de flux réseau.
2. **Implémenter et comparer deux approches RL** :
   - Q-Learning (sous une forme compatible avec des états continus, ex. DQN)
   - PPO (Proximal Policy Optimization)
3. **Vérifier expérimentalement une hypothèse théorique** sur la supériorité attendue de PPO dans ce contexte.

Ce roadmap documente les choix théoriques, le plan de rédaction du rapport et le plan de développement pour l’implémentation en Python.

---

## 2. Description du jeu de données CIC-DDoS2019

### 2.1 Contexte général

Le jeu de données **CIC-DDoS2019** a été produit par le *Canadian Institute for Cybersecurity* (Université du Nouveau-Brunswick).  
Il a pour objectif de fournir un **jeu de référence dédié à la détection d’attaques DDoS modernes**, en corrigeant les limites d’anciens jeux de données (KDD, NSL-KDD, CICIDS2017…) :

- Scénarios plus réalistes
- Plus grand nombre de types d’attaques
- Trafic de fond (bénin) plus varié

Le dataset contient à la fois :
- des **PCAP bruts** (captures réseau),
- et des **fichiers CSV de flux (flows)** déjà agrégés, étiquetés et enrichis de nombreuses caractéristiques via **CICFlowMeter-V3**.

Dans notre projet, nous travaillerons principalement sur les **flux agrégés** (fichiers CSV).

### 2.2 Scénario d’acquisition et environnement réseau

Le trafic est généré sur un banc d’essai contrôlé comprenant notamment :

- Un serveur web Ubuntu (IP différente selon le jour d’expérience),
- Un pare-feu matériel (Fortinet),
- Plusieurs postes clients Windows (7, Vista, 8.1, 10),
- Des scripts générant du trafic **bénin** (HTTP, HTTPS, FTP, SSH, e-mail…) via le système B-Profile, qui simule des utilisateurs humains.

Les attaques DDoS sont injectées par vagues à des **plages horaires précises** au milieu du trafic normal.  
Le dataset est structuré par **jours d’expériences**, chaque jour étant associé à un ensemble d’attaques :

- Jour 1 : PortMap, NetBIOS, LDAP, MSSQL, UDP, UDP-Lag, SYN, …
- Jour 2 : NTP, DNS, LDAP, MSSQL, NetBIOS, SNMP, SSDP, UDP, UDP-Lag, WebDDoS, SYN, TFTP, et PortScan (en test).

Cette organisation par jours sera utile pour définir des **épisodes RL** (un jour ou une portion de jour).

### 2.3 Types d’attaques

Les attaques sont regroupées dans une **taxonomie** large :

- **Attaques par réflexion** (reflection-based) : DNS, NTP, LDAP, MSSQL, NetBIOS, SNMP, SSDP, PortMap, TFTP, etc.
- **Attaques par exploitation** (exploitation-based) : SYN flood, UDP flood, UDP-Lag, WebDDoS, etc.

On retrouve ainsi **une douzaine de types d’attaques DDoS** distinctes, plus une classe BENIGN.  
Certains types d’attaques sont **très fréquents**, d’autres au contraire **extrêmement rares** (ex. WebDDoS).

### 2.4 Format des données

Les **fichiers CSV de flux** sont générés par CICFlowMeter-V3 à partir des PCAP.  
Chaque ligne représente un **flux réseau** (flow) caractérisé par :

- IP source / IP destination  
- Port source / port destination  
- Protocole  
- Features agrégées (statistiques)  
- Label (type d’attaque ou BENIGN)

### 2.5 Caractéristiques (features)

Le dataset comporte typiquement :

- **≈ 88 colonnes** :
  - ~87 caractéristiques numériques,
  - 1 colonne de label.

Les features couvrent notamment :

- Volume : nombre de paquets, octets envoyés/reçus,
- Durée et débit : durée du flux, bytes/s, packets/s,
- Statistiques sur les tailles de paquets (min, max, moyenne, écart-type),
- Informations sur les flags TCP,
- Inter-arrival times, ratios, etc.

Il s’agit donc d’un **espace d’états continu, de haute dimension**, ce qui a un impact direct sur le choix des algorithmes RL.

### 2.6 Taille et déséquilibres

Le dataset complet est très volumineux :

- Plus de **50 millions de flux** au total,
- Taille totale de l’ordre de plusieurs dizaines de Go,
- Distribué de manière **fortement déséquilibrée** entre types d’attaques et trafic bénin.

En pratique, de nombreux travaux n’utilisent pas le dataset complet, mais plutôt :

- des **sous-échantillons par classe** (par exemple ≤ 50 000 instances par type),
- ou un sous-ensemble de classes (par ex. BENIGN + {DNS, UDP, SYN}).

Ce déséquilibre important et ce volume massif sont deux contraintes majeures pour :
- la définition de la **récompense RL**,
- le choix des **métriques**,
- la **gestion des sous-ensembles** utilisés pour l’entraînement.

---

## 3. Hypothèse théorique : PPO vs Q-Learning

### 3.1 Contexte de la comparaison

Dans notre projet, l’environnement RL sera construit à partir de CIC-DDoS2019 avec :

- un **espace d’états continu et de haute dimension** (features de flux, éventuellement après réduction de dimension),
- un **espace d’actions discret** (ex. : prédire BENIGN vs ATTACK, ou plusieurs classes d’attaque),
- un **signal de récompense** lié à la qualité de la décision de détection,
- un **comportement temporel non stationnaire** (phases de trafic bénin, bursts d’attaques).

Nous allons comparer :

- **Q-Learning** (dans une version approchée par réseau de neurones si nécessaire, de type DQN),
- **PPO (Proximal Policy Optimization)**, algorithme de policy gradient / actor-critic.

### 3.2 Hypothèse formulée

> **H1 – Hypothèse principale :**  
> **PPO devrait théoriquement surpasser Q-Learning** pour la détection d’attaques DDoS sur CIC-DDoS2019, en termes de performances globales de détection et de stabilité d’apprentissage.

### 3.3 Justifications théoriques

1. **Gestion des états continus et complexes**  
   - Q-Learning tabulaire est inadapté lorsqu’il y a un grand nombre de features continues.
   - Même une version DQN reste plus fragile dans les environnements très bruités et à forte dimension.
   - PPO est conçu pour gérer des **politiques paramétrées** sur des espaces d’états continus et complexes.

2. **Robustesse à la non-stationnarité**  
   - Le trafic réseau évolue dans le temps (alternance périodes bénignes / attaques), ce qui rend l’environnement non stationnaire.
   - Les algorithmes de **policy gradient** (comme PPO) sont souvent plus robustes dans ces contextes que les méthodes purement basées sur la convergence d’une fonction de valeur.

3. **Environnements séquentiels bruités**  
   - La détection DDoS est un problème **séquentiel** : la nature du trafic à un instant dépend du contexte précédent.
   - PPO a été largement utilisé avec succès dans des environnements séquentiels complexes (jeux, contrôle continu, robots…).

4. **Stabilité de l’apprentissage**  
   - PPO introduit un mécanisme de **clipping** sur la mise à jour de la politique, améliorant la stabilité.
   - Q-Learning / DQN peut présenter des oscillations de convergence plus importantes, surtout dans des environnements riches et déséquilibrés.

Nous testerons cette hypothèse expérimentalement dans la suite du projet.

---

## 4. Axes de comparaison des deux modèles

Pour comparer Q-Learning et PPO, nous prévoyons plusieurs axes d’analyse.  
Ils seront utilisés dans l’implémentation puis repris dans le rapport.

### 4.1 Performances de classification

Sur un jeu de test (ou un jour entier non vu), nous mesurerons :

- Accuracy globale
- Précision (precision)
- Rappel (recall)
- F1-score
- MCC (Matthews Correlation Coefficient) pour le déséquilibre
- Matrices de confusion par classe

Un point d’attention sera mis sur les **attaques rares**, souvent critiques à détecter.

### 4.2 Stabilité et convergence

Nous étudierons les courbes :

- Reward moyen par épisode
- Variance des rewards
- Vitesse de convergence (nombre d’épisodes avant stabilisation)

Hypothèse implicite : PPO offre une **convergence plus lisse et plus stable** que Q-Learning/DQN.

### 4.3 Robustesse au déséquilibre

Nous évaluerons la sensibilité des modèles à :

- une forte majorité de trafic BENIGN,
- des sous-échantillonnages / sur-échantillonnages de certaines classes d’attaque.

Nous observerons notamment la variation de F1/MCC en fonction de la distribution des classes.

### 4.4 Sensibilité aux hyperparamètres

Nous comparerons la robustesse des modèles à des variations de :

- learning rate,
- gamma (facteur de discount),
- epsilon-greedy (pour Q-Learning/DQN),
- clipping range, batch size (pour PPO),
- architecture des réseaux (nombre de couches / neurones).

Un modèle plus stable vis-à-vis des hyperparamètres sera plus intéressant pour une utilisation pratique.

### 4.5 Coût computationnel

Nous comparerons :

- Temps d’entraînement total,
- Temps moyen par épisode,
- Consommation mémoire (RAM/GPU),
- Taille des modèles sauvegardés.

Q-Learning/DQN peut être plus simple mais pourrait devenir coûteux si la Q-function est complexe.

### 4.6 Résilience temporelle (généralisation)

Nous étudierons la **généralisation dans le temps** :

- Entraînement sur Jour 1 → test sur Jour 2
- Entraînement sur un sous-ensemble d’attaques → test sur un ensemble plus large

Cela permet de voir dans quelle mesure les modèles restent performants sur un trafic dont la distribution change.

---

## 5. Plan du rapport scientifique

Ce plan sera suivi pour le document final.

### I. Introduction

- Contexte général des attaques DDoS
- Limites des approches classiques de détection
- Motivation pour l’utilisation du RL
- Présentation des objectifs du projet
- Structure du rapport

### II. Description du jeu de données CIC-DDoS2019

- Scénario d’acquisition
- Types d’attaques
- Organisation temporelle (jours)
- Caractéristiques et volume des données
- Déséquilibre des classes
- Implications pour le RL

### III. Fondements théoriques

- Rappels sur l’apprentissage par renforcement (MDP, politique, valeur, récompense)
- Q-Learning / DQN : principe, forces, limites
- PPO : principe, notion de politique paramétrée, clipping, actor-critic
- Problématique de la détection DDoS formulée en RL

### IV. Hypothèse scientifique

- Formulation de l’hypothèse H1 sur la supériorité de PPO
- Justifications théoriques détaillées

### V. Méthodologie

- Prétraitement des données (nettoyage, réduction de dimension, normalisation)
- Construction de l’environnement RL (définition des états, actions, récompenses, épisodes)
- Implémentation de Q-Learning/DQN
- Implémentation de PPO
- Protocole expérimental : train/test, nombre d’épisodes, paramètres principaux

### VI. Résultats expérimentaux

- Performances de classification (métriques)
- Courbes de convergence
- Comparaison détaillée PPO vs Q-Learning
- Résultats par type d’attaque et par scénario

### VII. Discussion

- Analyse critique des résultats
- Validation ou non de l’hypothèse H1
- Limitations du travail (données, simplifications, ressources)
- Interprétation pour des cas réels

### VIII. Conclusion et perspectives

- Synthèse des contributions du projet
- Recommandations sur l’usage de PPO vs Q-Learning pour ce type de problème
- Pistes futures (architectures plus complexes, autres algos RL, systèmes temps réel)

---

## 6. Roadmap de développement Python (plan de dev)

Cette partie décrit le **plan de développement** du code Python.  
Elle pourra être reprise telle quelle dans le `README.md` technique du projet.

### Phase 0 – Préparation de l’environnement

- Créer la structure du dépôt (ex. `src/`, `data/`, `notebooks/`, `reports/`).
- Créer et activer un environnement virtuel.
- Installer les dépendances :
  - `numpy`, `pandas`, `scikit-learn`
  - `torch` (ou autre backend utilisé)
  - `gymnasium` (ou `gym`)
  - `stable-baselines3`
  - `matplotlib`, `seaborn` (pour les plots)
- Créer les fichiers de base :
  - `roadmap.md` (ce fichier)
  - `notes.md` (journal de bord scientifique / technique)
  - `README.md` (présentation générale du projet)

---

### Phase 1 – Prétraitement du dataset

**Objectif :** transformer CIC-DDoS2019 en un format exploitable comme environnement RL.

1. **Chargement des fichiers CSV**
   - Sélectionner un sous-ensemble (classes et taille) pour les premiers tests.
   - Gérer la concaténation et la structure temporelle si nécessaire.

2. **Nettoyage**
   - Retirer ou imputer les valeurs manquantes (NaN, inf).
   - Supprimer les colonnes non pertinentes (ID, timestamps si non utilisés, etc.).

3. **Réduction de dimension**
   - Option A : sélection de features (par importance via RandomForest, mutual information, etc.).
   - Option B : PCA pour obtenir un vecteur d’état de dimension réduite (par ex. 10–20 dimensions).

4. **Normalisation**
   - Appliquer un `StandardScaler` ou `MinMaxScaler` sur les features retenues.

5. **Construction séquentielle**
   - Option 1 : chaque flux = un état.
   - Option 2 : fenêtre glissante de plusieurs flux = un état (pour capturer plus de contexte).

6. **Sauvegarde intermédiaire**
   - Exporter les données prétraitées dans un format type `processed_dataset.pkl` pour accélérer les futurs essais.

---

### Phase 2 – Construction de l’environnement RL (Gym)

**Objectif :** encapsuler le dataset dans une classe d’environnement compatible Gym.

1. **Définir l’espace d’état (`observation_space`)**
   - Type : `Box` (vecteur continu).
   - Dimension : nombre de features après réduction.

2. **Définir l’espace d’action (`action_space`)**
   - Type : `Discrete(n)` :
     - n=2 pour BENIGN / ATTACK
     - ou n>2 pour plusieurs classes d’attaque.

3. **Définir la récompense**
   - Exemple simple :
     - +1 si prédiction correcte
     - −1 si prédiction incorrecte
   - Option : pénaliser davantage les faux négatifs (attaque prédite comme bénigne).

4. **Gestion du temps et des épisodes**
   - Un épisode peut correspondre à :
     - une portion séquentielle du dataset,
     - un jour complet (Jour 1, Jour 2),
     - ou une fenêtre temporelle fixée.
   - Implémenter `reset()` et `step()` en respectant la progression séquentielle dans les données.

5. **Implémentation**
   - Créer une classe `DDoSEnv(gym.Env)` dans `src/envs/ddos_env.py`.
   - Ajouter les contrôles pour éviter les erreurs (indices hors limites, etc.).

---

### Phase 3 – Implémentation de Q-Learning (version DQN)

**Objectif :** disposer d’une version fonctionnelle de Q-Learning adaptée aux états continus.

1. **Définir le réseau Q**
   - MLP simple (2–3 couches denses) avec activations ReLU.
   - Entrée : dimension de l’état.
   - Sortie : dimension = nombre d’actions.

2. **Implémenter l’agent DQN**
   - Stratégie epsilon-greedy.
   - Replay buffer.
   - Cible Q (target network) mise à jour périodiquement.

3. **Entraînement**
   - Boucle principale sur les épisodes :
     - interaction avec l’environnement,
     - stockage des transitions,
     - mises à jour du réseau Q.
   - Sauvegarder :
     - les récompenses par épisode,
     - le modèle entraîné.

4. **Évaluation**
   - Exécuter le modèle en mode exploitation (sans exploration).
   - Récupérer les actions, les comparaisons avec les labels réels, et calculer les métriques de classification.

---

### Phase 4 – Implémentation de PPO

**Objectif :** entraîner un agent PPO sur le même environnement.

1. **Intégration de PPO via Stable-Baselines3**
   - `from stable_baselines3 import PPO`
   - Utiliser la politique MLP par défaut (ou en définir une custom si nécessaire).

2. **Entraînement**
   - Définir les hyperparamètres de base (learning rate, gamma, batch size, clipping range, etc.).
   - Lancer l’entraînement sur un nombre d’étapes/épisodes défini.

3. **Logging**
   - Utiliser éventuellement TensorBoard pour suivre les récompenses.
   - Sauvegarder le modèle PPO entraîné.

4. **Évaluation**
   - De la même façon que pour DQN, exécuter le modèle en mode exploitation.
   - Calculer les mêmes métriques de classification pour comparaison.

---

### Phase 5 – Comparaison et analyse expérimentale

**Objectif :** appliquer les axes de comparaison définis en section 4.

1. **Protocole commun**
   - Préparer un scénario d’évaluation identique pour DQN et PPO :
     - même sous-ensemble de données,
     - même split temporel (train/test),
     - mêmes métriques de sortie.

2. **Collecte des métriques**
   - Accuracy, precision, recall, F1, MCC.
   - Matrices de confusion.
   - Courbes de reward moyen par épisode.
   - Courbes de convergence (reward, loss).

3. **Tests de robustesse**
   - Varier la distribution des classes.
   - Varier quelques hyperparamètres clés.

4. **Synthèse des résultats**
   - Comparer DQN vs PPO sur tous les axes :
     - performances de détection,
     - stabilité,
     - coût de calcul,
     - généralisation.

---

### Phase 6 – Rédaction du rapport

**Objectif :** transformer le travail de dev + résultats en un rapport structuré.

1. Compléter les sections du rapport (plan en section 5).
2. Intégrer les figures :
   - Courbes d’apprentissage,
   - Matrices de confusion,
   - Tableaux de résultats.
3. Rédiger la discussion :
   - Validation ou non de l’hypothèse H1,
   - Interprétation des différences PPO vs Q-Learning.
4. Rédiger la conclusion et les perspectives.

---

## 7. Liens avec notes.md / README.md

- `notes.md` : servira à détailler chaque phase (journal de bord, détails techniques, choix, erreurs, ajustements).
- `README.md` : présentera :
  - le contexte du projet,
  - le but général,
  - les prérequis techniques,
  - un guide rapide pour exécuter les expériences,
  - un résumé des phases de cette roadmap (mais sans tout le détail analytique).

Ce `roadmap.md` joue le rôle de **document de référence structurant** :  
il connecte les aspects théoriques (hypothèse, axes de comparaison), scientifiques (plan de rapport) et techniques (plan de développement Python).