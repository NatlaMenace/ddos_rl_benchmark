# 📘 **Détection d’attaques DDoS par Apprentissage par Renforcement (PPO vs Q-Learning)**

---

## 🔷 1. Introduction

Ce projet vise à comparer l’efficacité de deux algorithmes d’apprentissage par renforcement (RL) – **PPO (Proximal Policy Optimization)** et **Q-Learning** – pour la détection d’attaques DDoS dans un environnement simulé de réseau.

---

## 🔷 2. Objectifs

- Concevoir un environnement simulant des attaques DDoS.
- Implémenter et entraîner des agents RL avec PPO et Q-Learning.
- Comparer leurs performances en termes de détection, précision et temps d’apprentissage.

---

## 🔷 3. Prérequis

- Python 3.8+
- pip

---

## 🔷 Dataset CIC-DDoS2019

Le projet utilise le dataset CIC-DDoS2019 (Canadian Institute for Cybersecurity).
Le téléchargement se fait automatiquement via kagglehub :

python -m src.data.download_cicddos2019

---

## 🔷 4. Installation

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/votre-utilisateur/ddos_rl_benchmark.git
   cd ddos_rl_benchmark
   ```
2. **Créer un environnement virtuel (optionnel mais recommandé) :**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```
3. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```
4. **Télécharger le dataset :**
   ```bash
   python -m src.data.download_cicddos2019
   ```

---

## 🔷 5. Structure du projet

```
ddos_rl_benchmark/
│── data/
│   └── raw/
│── src/
│   ├── agents/
│   ├── envs/
│   └── data/
│── main.py
│── notes.md
│── README.md
│── requirements.txt
```

*Note : Avant d’entraîner les agents, il est nécessaire de prétraiter les données.*

### 🔷 Phase 2 — Prétraitement

Avant l’entraînement des agents, lancer le pipeline de prétraitement :
```
python -m src.data.preprocessing
```
Cela génère automatiquement les fichiers normalisés dans `data/processed/`.

### 🔷 Phase 3 — Baseline supervisée

Pour entraîner la baseline supervisée (RandomForest) :

```bash
python -m src.agents.baseline_supervised
```

Ce script génère automatiquement :

- `reports/baseline_report.md` — rapport lisible en Markdown
- `reports/confusion_matrix.png` — heatmap de la matrice de confusion
- `data/processed/baseline_random_forest.joblib` — modèle sauvegardé

L’environnement RL utilisé par les futurs agents (Q-Learning et PPO) est défini dans :

```
src/envs/ddos_env.py
```

---

### 🔷 Phase 4 — Q-Learning (DQN)

Entraîner l’agent DQN (version Deep Q-Learning) sur l’environnement DDoS :

Exemple d’entraînement :
```bash
python main_train_dqn.py --episodes 200 --device cpu --split train --out-dir models/dqn
```

Les modèles et courbes d’entraînement sont sauvegardés dans :

```
models/dqn/
```

*Note : PPO sera ajouté en Phase 5. Les commandes PPO dans la section “Utilisation” seront activées une fois cette phase complétée.*

---

### 🔷 Phase 5 — PPO (Policy Gradient)

Entraîner l’agent PPO (Stable-Baselines3) sur le même environnement DDoS :

```bash
python main_train_ppo.py --total-timesteps 500000 --device cpu --max-steps 1000
```

Les modèles et rapports générés sont sauvegardés dans :

```
models/ppo/
reports/
```

- `models/ppo/ppo_cicddos.zip` : modèle PPO entraîné
- `reports/ppo_report.md` : rapport de classification sur le test
- `reports/ppo_confusion_matrix.png` : matrice de confusion PPO
```

---

## 🔷 6. Utilisation

⚠️ *Cette section sera mise à jour lorsque les scripts d’entraînement
(PPO et Q-Learning) seront finalisés.*

Les commandes ci-dessous sont indicatives et seront ajustées :

### Lancer une expérience PPO :
```bash
python main.py --algo ppo --episodes 1000
```

### Lancer une expérience Q-Learning :
```bash
python main.py --algo qlearning --episodes 1000
```

### Options principales :
- `--algo` : Choix de l’algorithme (`ppo` ou `qlearning`)
- `--episodes` : Nombre d’épisodes d’entraînement
- `--render` : Affiche l’environnement (si applicable)

---

## 🔷 7. Résultats attendus

- **Courbes d’apprentissage** : Précision, taux de détection, taux de faux positifs.
- **Comparaison** : Tableaux comparatifs entre PPO et Q-Learning.
- **Reproductibilité** : Scripts et seeds pour répéter les expériences.

---

## 🔷 8. Références

- [OpenAI Gym](https://gym.openai.com/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Introduction to Reinforcement Learning (Sutton & Barto)](http://incompleteideas.net/book/the-book.html)

---

## 🔷 9. Auteurs

- **Nathan Hérault** – UQO
- **Bafodé Koulibaly** – UQO

---

## 🔷 10. Licence

Ce projet est sous licence MIT.