### Pourquoi recourir au DQN ?

Dans notre projet, le Q-Learning tabulaire classique est inadapté, car il repose sur une Q-table ne pouvant gérer qu’un espace d’états discret et de faible dimension. Or, le dataset CIC-DDoS2019 produit des états continus à haute dimension, rendant impossible toute représentation tabulaire. Nous utilisons donc le Deep Q-Learning (DQN), dans lequel la Q-table est remplacée par un réseau de neurones approximant la fonction de valeur d’action Q(s,a). Cette fonction Q représente la qualité attendue d’une action dans un état donné et permet à l’agent d’orienter sa politique en choisissant les actions maximisant cette valeur. DQN constitue ainsi une condition nécessaire pour appliquer le Q-Learning dans un environnement continu et complexe comme celui de la détection d’attaques DDoS.

### 📌 Phase 1 – Prétraitement du dataset : Résumé et choix méthodologiques

Cette première phase avait pour objectif de transformer le dataset CIC-DDoS2019 en un format exploitable par un environnement d’apprentissage par renforcement. Elle constitue la base de tout le pipeline RL, garantissant cohérence, stabilité et reproductibilité des expériences ultérieures (PPO et Q-Learning).

⸻

1. Chargement et préparation initiale

Les fichiers du dataset ont été téléchargés et chargés automatiquement depuis data/raw/cicddos2019/.
Plusieurs points clés ont été mis en place :
	•	Concaténation multi-fichiers (training/testing, variations UDP/TCP/Benign).
	•	Sélection d’un sous-ensemble pour les premiers essais (contrôle du volume).
	•	Préservation de l’ordre temporel lorsque disponible (important pour un modèle séquentiel RL).

Ce prétraitement unifié garantit une base cohérente malgré la structure hétérogène d’origine du dataset.

⸻

2. Nettoyage des données

Un nettoyage systématique a été effectué :
	•	Remplacement des valeurs inf par NaN, puis imputation à la médiane (stratégie robuste aux distributions asymétriques fréquentes en trafic réseau).
	•	Suppression des colonnes non pertinentes : identifiants, métadonnées, colonnes constantes, timestamps inutilisés.
	•	Préservation explicite de la colonne Label, même lorsqu’elle est temporairement constante dans un sous-échantillon.

Ce nettoyage assure un dataset pleinement numérique et exploitable pour les méthodes de sélection et de normalisation.

⸻

3. Réduction de dimension

Deux approches étaient envisageables : PCA ou sélection de features.
Après analyse méthodologique, nous avons retenu :

➤ Option choisie : Sélection supervisée de features
	•	Méthodes utilisées :
RandomForest feature importance + Mutual Information (combinaison pondérée).
	•	Justifications :
	•	Interprétabilité plus forte que PCA.
	•	Stabilité supérieure pour PPO et DQN.
	•	Alignement avec les pratiques en sécurité réseau.
	•	Maintien du sens physique des features (ex. Flow Duration, Total Fwd Packets…).

Le top-k final (k = 20) constitue la base de l’état dans l’environnement RL.

⸻

4. Normalisation

Les features sélectionnées ont été normalisées via un StandardScaler, puis le scaler a été sauvegardé pour garantir la reproductibilité des entraînements.

Choix justifié par :
	•	meilleure convergence des algorithmes de type policy gradient (PPO),
	•	caractéristiques du trafic réseau présentant des amplitudes très différentes.

⸻

5. Construction séquentielle (structure de l’état RL)

Deux représentations possibles ont été étudiées :
	•	Un flux = un état
	•	Fenêtre glissante de flux = un état

➤ Option retenue : fenêtre glissante
	•	Paramètre choisi : window_size = 32
	•	Motifs :
	•	capture de la dynamique temporelle d’un DDoS,
	•	stabilité accrue pour PPO,
	•	cohérence avec la littérature scientifique RL + cybersécurité,
	•	observation suffisamment riche sans être trop dimensionnelle.

L’état final = concaténation flattenée de 32×20 valeurs scalées.

⸻

6. Sauvegarde intermédiaire

Pour accélérer les expérimentations, le dataset final prétraité a été exporté sous :
data/processed/processed_dataset.pkl

ainsi que :
	•	selected_features.json
	•	scaler.pkl

Cette étape permet de relancer des entraînements RL sans repasser par les étapes lourdes de prétraitement.

⸻

✔️ Conclusion de la Phase 1

Grâce à cette phase, nous avons obtenu :
	•	un dataset nettoyé, réduit, normalisé, séquentiel,
	•	une représentation d’état cohérente pour PPO et Q-Learning,
	•	un pipeline reproductible, modulaire et optimisé,
	•	un format final directement utilisable par l’environnement Gym personnalisé.

La Phase 1 constitue ainsi un socle méthodologique solide pour la comparaison expérimentale PPO vs Q-Learning.


Phase 4 – Synthèse comparative DQN vs PPO

Performances de détection

Les deux agents, entraînés et évalués selon un protocole strictement identique, atteignent une détection extrêmement efficace des attaques (recall > 99%). Toutefois, leurs comportements divergent fortement concernant la classification des flux bénins. PPO adopte une stratégie fortement biaisée vers la prédiction Attack, ce qui maximise la détection d’attaques mais au prix d’un taux très élevé de faux positifs. À l’inverse, DQN parvient à identifier une proportion significative de trafic bénin tout en conservant d’excellentes performances sur la détection d’attaques.

Stabilité d’apprentissage

Les courbes issues de TensorBoard montrent une convergence plus régulière pour PPO, caractéristique de l’approche actor-critic. Les pertes et récompenses évoluent de manière plus stable. DQN, en revanche, présente des oscillations importantes tant dans la loss que dans la reward, reflétant la difficulté du Q-learning dans cet espace d’état compressé et séquentiel.

Coût de calcul

PPO est plus coûteux en temps et en ressources, du fait de ses multiples passes d’optimisation et de l’entraînement simultané d’un acteur et d’un critique. DQN, reposant sur un MLP unique et un mécanisme de replay buffer, est plus léger et plus rapide.

Généralisation

Sur le dataset de test, DQN montre une meilleure capacité à généraliser en équilibrant la détection du trafic légitime et malveillant. PPO, bien que performant pour identifier les attaques, peine à reconnaître les flux bénins.

Conclusion

Dans le cadre de la détection d’attaques DDoS sur CIC-DDoS2019, DQN obtient les meilleures performances globales, tandis que PPO se démarque par une stabilité d’entraînement supérieure. Le choix dépend des objectifs opérationnels : maximiser la détection d’attaques (PPO) ou réduire les faux positifs en conservant un haut niveau de détection (DQN).