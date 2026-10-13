# Catalogue des sujets 2026-2027 — IA probabiliste, théorie des jeux et machine learning

Ce catalogue est propre à la promotion **MSMIS5IN11 2026-2027 (Ingénierie et santé)**. Tous les sujets ont été vérifiés contre les projets réalisés depuis 2023 à l'EPF, à l'ECE et à l'EPITA (voir [l'historique](HISTORIQUE-SUJETS.md)) : aucun ne reprend le verrou technique d'un projet déjà rendu. La plupart portent sur la santé ; la catégorie **F** propose quelques sujets hors santé pour les groupes qui le souhaitent.

Chaque fiche donne une difficulté indicative, un **MVP** (ce qui doit fonctionner le jour de la soutenance), un critère de **réussite**, des **extensions** possibles et une analyse de **faisabilité** avec un plan B. Les liens **CoursIA** pointent vers les notebooks du dépôt [jsboige/CoursIA](https://github.com/jsboige/CoursIA) qui servent de point de départ.

> **Données de santé.** Utilisez uniquement des jeux de données publics en accès ouvert, ou des données synthétiques. Aucune donnée réelle de patient, même « anonymisée » par vos soins, ne doit être commitée. Les jeux à accès contrôlé (MIMIC, UK Biobank…) sont exclus faute de temps pour obtenir les accréditations.
>
> **Pas de diagnostic réel.** Les démonstrateurs sont des outils pédagogiques : ils ne fournissent aucun avis médical et doivent le dire.

## Index

| Code | Sujet | Difficulté |
|---|---|:---:|
| **A** | **Inférence bayésienne et programmation probabiliste** | |
| A1 | Prédiction de glycémie par modèle d'état bayésien | 3/5 |
| A2 | Analyse de survie bayésienne hiérarchique multi-centres | 3/5 |
| A3 | Agréger les avis de plusieurs radiologues : modèle de Dawid-Skene | 3/5 |
| A4 | Détection bayésienne de ruptures dans les signaux physiologiques | 3/5 |
| A5 | Méta-analyse bayésienne d'essais cliniques | 2/5 |
| A6 | Psychométrie bayésienne d'un questionnaire de santé (IRT) | 3/5 |
| **B** | **Décision sous incertitude et causalité** | |
| B1 | Effet causal d'un traitement à partir de données observationnelles | 3/5 |
| B2 | Valeur de l'information d'un examen complémentaire | 3/5 |
| B3 | Essai clinique adaptatif par bandits | 3/5 |
| B4 | Surveillance ou traitement ? Un POMDP de suivi thérapeutique | 4/5 |
| B5 | Équité et calibration d'un score de risque de réhospitalisation | 3/5 |
| **C** | **Théorie des jeux et mécanismes en santé** | |
| C1 | Le jeu de la vaccination : passager clandestin et immunité collective | 3/5 |
| C2 | L'antibiorésistance comme tragédie des communs répétée | 3/5 |
| C3 | Partager le coût d'un plateau technique mutualisé (jeux coopératifs) | 3/5 |
| C4 | Jeux de sécurité de Stackelberg pour les inspections sanitaires | 3/5 |
| C5 | Persuasion bayésienne : communiquer un risque sanitaire | 4/5 |
| C6 | Antisélection et contrats d'assurance santé | 3/5 |
| **D** | **Machine learning et deep learning sur données de santé** | |
| D1 | Classification d'ECG calibrée et sélective | 3/5 |
| D2 | Stadification automatique du sommeil | 3/5 |
| D3 | Apprentissage fédéré entre hôpitaux simulés | 4/5 |
| D4 | Robustesse au changement d'hôpital en imagerie médicale | 4/5 |
| D5 | Alerte précoce de sepsis en réanimation | 3/5 |
| **E** | **IA générative appliquée à la santé** | |
| E1 | Du texte libre au réseau bayésien : orientation de symptômes avec abstention | 3/5 |
| E2 | Vérification d'affirmations de santé avec citations | 3/5 |
| E3 | Données de patients synthétiques : fidélité, utilité et confidentialité | 3/5 |
| **F** | **Hors santé (débordement)** | |
| F1 | Pragmatique bayésienne : le modèle RSA | 4/5 |
| F2 | Jeux à champ moyen : évacuation d'une foule | 4/5 |

---

## A — Inférence bayésienne et programmation probabiliste

### A1 — Prédiction de glycémie par modèle d'état bayésien
**Difficulté : 3/5** · Mots-clés : `state-space`, `filtre de Kalman`, `diabète de type 1`, `PyMC`

Un patient diabétique de type 1 porte un capteur de glucose en continu. Le groupe modélise la glycémie comme un état caché influencé par les repas et l'insuline, puis prédit la valeur dans 30 à 60 minutes, avec un intervalle de crédibilité.
- **MVP** : simulateur open source (`simglucose`, modèle UVA/Padova) → modèle d'état linéaire-gaussien filtré, prédiction à 30 min avec intervalle.
- **Réussite** : RMSE et couverture des intervalles comparées à une baseline naïve (dernière valeur, régression linéaire) sur plusieurs patients virtuels ; analyse des hypoglycémies manquées.
- **Extensions** : modèle non linéaire (filtre particulaire), effets patient hiérarchiques, alerte d'hypoglycémie avec coût asymétrique.
- **Faisabilité** : tout est simulé, aucun accès à des données réelles. Plan B : filtre de Kalman analytique (NumPy) si l'inférence MCMC est trop lente.

**CoursIA** : [PyMC-17-Kalman-Filter](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-17-Kalman-Filter.ipynb) · [Infer-17-Kalman-Filter](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-17-Kalman-Filter.ipynb)
**Références** : [simglucose](https://github.com/jxx123/simglucose) · Man et al., *The UVA/PADOVA Type 1 Diabetes Simulator*, J. Diabetes Sci. Technol., 2014.

### A2 — Analyse de survie bayésienne hiérarchique multi-centres
**Difficulté : 3/5** · Mots-clés : `survie`, `Weibull`, `censure`, `modèle hiérarchique`

Les durées de survie après un traitement varient d'un centre à l'autre. Le groupe estime un modèle de survie bayésien qui partage l'information entre centres et quantifie l'incertitude sur l'effet du traitement.
- **MVP** : jeu public (`scikit-survival` : GBSG2, WHAS500 ou Veterans lung cancer) → Kaplan-Meier, puis modèle de Weibull bayésien avec covariables et censure à droite.
- **Réussite** : comparaison avec un modèle de Cox fréquentiste (`lifelines`) ; vérification prédictive a posteriori ; courbes de survie individuelles avec bandes de crédibilité.
- **Extensions** : effets aléatoires par centre ou sous-groupe, modèle à risques compétitifs, sélection de modèle (LOO).
- **Faisabilité** : jeux de quelques centaines de lignes, PyMC suffit sur CPU. Plan B : modèle exponentiel si la Weibull converge mal.

**CoursIA** : [PyMC-19-Survival-Analysis](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-19-Survival-Analysis.ipynb) · [PyMC-12-Modeles-Hierarchiques](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-12-Modeles-Hierarchiques.ipynb)
**Références** : [scikit-survival datasets](https://scikit-survival.readthedocs.io/en/stable/api/datasets.html) · Ibrahim, Chen & Sinha, *Bayesian Survival Analysis*, Springer, 2001.

### A3 — Agréger les avis de plusieurs radiologues : modèle de Dawid-Skene
**Difficulté : 3/5** · Mots-clés : `crowdsourcing`, `variables latentes`, `accord inter-annotateurs`

Quand plusieurs radiologues notent la même lésion, ils ne sont pas d'accord. Le groupe infère conjointement la « vraie » étiquette de chaque cas et la fiabilité de chaque annotateur, plutôt que de prendre un vote majoritaire.
- **MVP** : notes de malignité des nodules pulmonaires LIDC-IDRI (4 radiologues, via `pylidc`) ou jeu synthétique calibré → modèle de Dawid-Skene bayésien.
- **Réussite** : comparaison avec le vote majoritaire et le kappa de Fleiss ; matrices de confusion par annotateur ; analyse des cas les plus incertains.
- **Extensions** : difficulté propre à chaque cas, apprentissage actif (quel cas faire relire ?), annotateurs adverses.
- **Faisabilité** : seules les annotations sont nécessaires, pas les images. Plan B : données synthétiques avec annotateurs de fiabilité connue, ce qui permet de vérifier la récupération des paramètres.

**CoursIA** : [Infer-13-Crowdsourcing](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-13-Crowdsourcing.ipynb) · [PyMC-13-Crowdsourcing](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-13-Crowdsourcing.ipynb)
**Références** : Dawid & Skene, *Maximum Likelihood Estimation of Observer Error-Rates*, JRSS C, 1979 · [pylidc](https://pylidc.github.io/).

### A4 — Détection bayésienne de ruptures dans les signaux physiologiques
**Difficulté : 3/5** · Mots-clés : `change-point`, `BOCPD`, `variabilité cardiaque`, `PhysioNet`

Un signal physiologique (fréquence cardiaque, respiration) change brusquement de régime lors d'un événement clinique. Le groupe détecte ces ruptures en ligne, avec une probabilité associée à chaque alerte.
- **MVP** : enregistrements publics PhysioNet (par ex. *MIT-BIH Atrial Fibrillation* pour les épisodes de FA) → détection bayésienne de ruptures en ligne (Adams & MacKay) sur l'intervalle RR.
- **Réussite** : précision/rappel des ruptures détectées contre les annotations, délai de détection, comparaison avec CUSUM ou un seuil fixe.
- **Extensions** : modèle hors ligne avec nombre de ruptures inconnu, signaux multivariés, coût des fausses alarmes.
- **Faisabilité** : données téléchargeables via `wfdb`. Plan B : séries simulées à ruptures connues.

**CoursIA** : [PyMC-18-Change-Point](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-18-Change-Point.ipynb) · [Infer-18-Change-Point](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-18-Change-Point.ipynb)
**Références** : Adams & MacKay, *Bayesian Online Changepoint Detection*, arXiv:0710.3742, 2007 · [PhysioNet](https://physionet.org/).

### A5 — Méta-analyse bayésienne d'essais cliniques
**Difficulté : 2/5** · Mots-clés : `méta-analyse`, `effets aléatoires`, `hétérogénéité`, `shrinkage`

Plusieurs essais testent le même traitement avec des résultats divergents. Le groupe combine leurs résultats dans un modèle hiérarchique bayésien, puis mesure l'hétérogénéité entre études.
- **MVP** : une méta-analyse publiée (jeux du paquet R `metadat`, exportés en CSV) → modèle à effets aléatoires sous PyMC, *forest plot* a posteriori.
- **Réussite** : comparaison avec l'estimateur fréquentiste de DerSimonian-Laird ; sensibilité aux priors sur l'hétérogénéité ; distribution prédictive pour un nouvel essai.
- **Extensions** : biais de publication, méta-régression sur une covariable, méta-analyse en réseau de plusieurs traitements.
- **Faisabilité** : petits jeux, sujet accessible. Plan B : modèle à effet fixe comme référence.

**CoursIA** : [PyMC-12-Modeles-Hierarchiques](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-12-Modeles-Hierarchiques.ipynb) · [PyMC-10-Model-Selection](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-10-Model-Selection.ipynb)
**Références** : [metadat](https://wviechtb.github.io/metadat/) · Higgins, Thompson & Spiegelhalter, *A re-evaluation of random-effects meta-analysis*, JRSS A, 2009.

### A6 — Psychométrie bayésienne d'un questionnaire de santé (IRT)
**Difficulté : 3/5** · Mots-clés : `théorie de la réponse à l'item`, `PHQ-9`, `variables latentes`

Un questionnaire de dépistage de la dépression additionne des items, mais tous ne mesurent pas aussi bien la sévérité. Le groupe estime un modèle IRT bayésien pour placer chaque répondant et chaque item sur une échelle latente.
- **MVP** : réponses PHQ-9 du NHANES (public) → modèle IRT gradué (*graded response*) sous PyMC.
- **Réussite** : paramètres de discrimination et de seuil par item, comparaison du score latent avec le score additif, fonctions d'information.
- **Extensions** : fonctionnement différentiel des items (DIF) entre sous-groupes, questionnaire adaptatif qui choisit l'item suivant.
- **Faisabilité** : NHANES est téléchargeable librement (fichiers XPT). Plan B : modèle 2PL sur items dichotomisés.

**CoursIA** : [PyMC-07-Skills-IRT](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-07-Skills-IRT.ipynb) · [Infer-7-Skills-IRT](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Infer/Infer-7-Skills-IRT.ipynb)
**Références** : [NHANES](https://wwwn.cdc.gov/nchs/nhanes/) · Samejima, *Estimation of latent ability using a response pattern of graded scores*, 1969.

---

## B — Décision sous incertitude et causalité

### B1 — Effet causal d'un traitement à partir de données observationnelles
**Difficulté : 3/5** · Mots-clés : `inférence causale`, `DoWhy`, `score de propension`, `confusion`

Les patients traités ne sont pas comparables aux non-traités. Le groupe estime l'effet causal moyen d'une intervention médicale en explicitant ses hypothèses sous forme de graphe causal, puis éprouve leur robustesse.
- **MVP** : jeu *Right Heart Catheterization* (SUPPORT, public) ou IHDP semi-synthétique → graphe causal, identification, estimation par appariement et pondération (IPW) avec DoWhy.
- **Réussite** : comparaison avec l'estimation naïve ; au moins deux tests de réfutation (placebo, confondant non observé) ; discussion clinique des hypothèses.
- **Extensions** : effets hétérogènes (EconML, *causal forests*), variable instrumentale, découverte de structure.
- **Faisabilité** : DoWhy est mûr et bien documenté. Plan B : données semi-synthétiques à effet connu pour valider la méthode.

**CoursIA** : [CausalBridges-02-Dowhy-Estimand-Intervention](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/Causal-Bridges/CausalBridges-02-Dowhy-Estimand-Intervention.ipynb) · [CausalBridges-05-Dowhy-Sensibilite-Confounder](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/Causal-Bridges/CausalBridges-05-Dowhy-Sensibilite-Confounder.ipynb) · [PyMC-05-Causal-Inference](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-05-Causal-Inference.ipynb)
**Références** : [DoWhy](https://www.pywhy.org/dowhy/) · Hernán & Robins, *Causal Inference: What If*, 2020 (en accès libre).

### B2 — Valeur de l'information d'un examen complémentaire
**Difficulté : 3/5** · Mots-clés : `réseau de décision`, `diagramme d'influence`, `valeur de l'information`

Faut-il prescrire un examen supplémentaire avant de décider d'un traitement ? Le groupe construit un réseau de décision qui chiffre ce que l'examen apporte, compte tenu de sa sensibilité, de sa spécificité et de son coût.
- **MVP** : un cas clinique documenté (par ex. suspicion d'embolie pulmonaire : D-dimères puis scanner) → réseau bayésien (`pgmpy`) + nœuds de décision et d'utilité, calcul de la valeur de l'information parfaite et imparfaite.
- **Réussite** : stratégie optimale selon la probabilité *a priori* ; analyse de sensibilité (seuils de décision) ; comparaison avec une règle clinique publiée.
- **Extensions** : séquence de plusieurs examens, utilités en QALY, préférences du patient.
- **Faisabilité** : modèle de petite taille, paramètres tirés de la littérature. Plan B : arbre de décision évalué à la main.

**CoursIA** : [DecInfer-05-Decision-Networks](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/DecInfer/DecInfer-05-Decision-Networks.ipynb) · [DecInfer-06-Value-Information](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/DecInfer/DecInfer-06-Value-Information.ipynb) · [DecPyMC-5-Value-Information](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/DecPyMC/DecPyMC-5-Value-Information.ipynb)
**Références** : Kochenderfer et al., *Algorithms for Decision Making*, MIT Press, 2022 (bibliothèque du cours) · Pauker & Kassirer, *The threshold approach to clinical decision making*, NEJM, 1980.

### B3 — Essai clinique adaptatif par bandits
**Difficulté : 3/5** · Mots-clés : `bandits`, `Thompson sampling`, `randomisation adaptative`, `éthique`

Dans un essai classique, la moitié des patients reçoit le traitement le moins bon jusqu'à la fin. Un essai adaptatif réalloue les patients au fil des résultats. Le groupe simule et compare ces stratégies.
- **MVP** : simulateur d'essai à plusieurs bras (taux de réponse paramétrables) → randomisation fixe vs Thompson sampling vs UCB.
- **Réussite** : nombre de patients bien traités, puissance statistique finale et erreur de type I mesurées sur des milliers de simulations.
- **Extensions** : arrêt précoce bayésien, réponse retardée, dérive temporelle, cas réel rejoué (essai ECMO, controverse historique).
- **Faisabilité** : simulation pure, peu de calcul. Plan B : deux bras seulement.

**CoursIA** : [DecInfer-10-Thompson-Sampling](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/DecInfer/DecInfer-10-Thompson-Sampling.ipynb) · [RL-04-Bandits-Manchots](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/RL/RL-04-Bandits-Manchots-Python.ipynb)
**Références** : Villar, Bowden & Wason, *Multi-armed Bandit Models for the Optimal Design of Clinical Trials*, Statistical Science, 2015.

### B4 — Surveillance ou traitement ? Un POMDP de suivi thérapeutique
**Difficulté : 4/5** · Mots-clés : `POMDP`, `croyances`, `dépistage`, `planification séquentielle`

L'état réel d'une maladie n'est jamais observé directement, seulement à travers des examens imparfaits. Le groupe formule le suivi d'un patient (surveiller, examiner, traiter) comme un POMDP et calcule une politique.
- **MVP** : modèle de progression à 3–4 états inspiré de la littérature (par ex. surveillance d'un nodule ou d'un anévrisme), observations bruitées → mise à jour de croyance et politique par itération sur la valeur (point-based).
- **Réussite** : comparaison avec des protocoles fixes (examen tous les 6 ou 12 mois) sur coût, détection tardive et nombre d'examens.
- **Extensions** : paramètres incertains (POMDP bayésien), apprentissage par renforcement, explication de la politique.
- **Faisabilité** : le modèle doit rester petit. Plan B : MDP entièrement observable comme référence.

**CoursIA** : [RL-11-POMDP-Croyances](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/RL/RL-11-POMDP-Croyances-Python.ipynb) · [DecInfer-08-Sequential](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/DecInfer/DecInfer-08-Sequential.ipynb)
**Références** : Steimle & Denton, *Markov Decision Processes for Screening and Treatment of Chronic Diseases*, 2017 · Kochenderfer et al., *Algorithms for Decision Making*, ch. 19-22.

### B5 — Équité et calibration d'un score de risque de réhospitalisation
**Difficulté : 3/5** · Mots-clés : `fairness`, `calibration`, `score de risque`, `biais`

Un score qui prédit la réhospitalisation peut être bien calibré en moyenne et mal calibré pour certains groupes. Le groupe entraîne un tel score, audite son équité, puis évalue des corrections.
- **MVP** : jeu *Diabetes 130-US hospitals* (UCI) → modèle de réhospitalisation à 30 jours, calibration globale et par sous-groupe, métriques d'équité (`fairlearn`).
- **Réussite** : diagnostic chiffré des écarts ; au moins une correction (recalibration par groupe, contrainte d'équité) avec son coût en performance ; discussion des notions d'équité incompatibles.
- **Extensions** : lecture causale de l'équité (chemins de discrimination), incertitude bayésienne sur les métriques.
- **Faisabilité** : 100 000 lignes tabulaires, CPU suffisant. Plan B : régression logistique seule.

**CoursIA** : [CausalBridges-08-Causal-Fairness](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/Causal-Bridges/CausalBridges-08-Causal-Fairness.ipynb) · [ML-04-Evaluation-Python](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/ML.Net/ML-04-Evaluation-Python.ipynb)
**Références** : [Diabetes 130-US hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) · Obermeyer et al., *Dissecting racial bias in an algorithm used to manage the health of populations*, Science, 2019.

---

## C — Théorie des jeux et mécanismes en santé

### C1 — Le jeu de la vaccination : passager clandestin et immunité collective
**Difficulté : 3/5** · Mots-clés : `équilibre de Nash`, `optimum social`, `externalités`, `jeu de population`

Chaque individu décide de se vacciner en comparant le risque perçu du vaccin et celui de la maladie, alors que la protection dépend aussi de ce que font les autres. Le groupe calcule l'équilibre, le compare à l'optimum collectif et teste des politiques publiques.
- **MVP** : jeu de population couplé à un modèle SIR à l'équilibre → couverture vaccinale d'équilibre vs couverture socialement optimale, en fonction du risque perçu.
- **Réussite** : courbes d'équilibre et analyse de l'écart ; effet chiffré d'au moins deux leviers (subvention, obligation, information).
- **Extensions** : dynamique d'imitation (réplicateur), hétérogénéité des risques, communication sur les effets indésirables.
- **Faisabilité** : modèles analytiques et simulation légère. **Différenciation** : une simulation épidémique multi-agents a déjà été réalisée en 2025 ; ici, le verrou est l'**équilibre stratégique** des décisions individuelles, pas la propagation.

**CoursIA** : [GameTheory-04-NashEquilibrium](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-04-NashEquilibrium-Python.ipynb) · [GameTheory-06-EvolutionTrust](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-06-EvolutionTrust-Python.ipynb)
**Références** : Bauch & Earn, *Vaccination and the theory of games*, PNAS, 2004.

### C2 — L'antibiorésistance comme tragédie des communs répétée
**Difficulté : 3/5** · Mots-clés : `jeux répétés`, `biens communs`, `coopération`, `prescription`

Chaque prescripteur a intérêt à prescrire un antibiotique à large spectre, mais l'efficacité collective s'érode. Le groupe modélise ce dilemme comme un jeu répété dont la ressource commune s'épuise.
- **MVP** : jeu répété à N prescripteurs avec stock d'efficacité qui décroît avec l'usage → tournoi de stratégies (égoïste, réciprocité, seuil).
- **Réussite** : conditions de coopération (facteur d'actualisation, observabilité) ; comparaison avec l'optimum social ; effet d'une régulation (quota, taxe, surveillance).
- **Extensions** : apprentissage multi-agents, hôpitaux de tailles différentes, information imparfaite sur les prescriptions des autres.
- **Faisabilité** : simulation simple. Plan B : jeu à deux joueurs avec résolution analytique.

**CoursIA** : [GameTheory-06c-RepeatedGames-FolkTheorem](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-06c-RepeatedGames-FolkTheorem-Python.ipynb) · [GameTheory-17-MultiAgent-RL](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-17-MultiAgent-RL-Python.ipynb)
**Références** : Laxminarayan & Brown, *Economics of antibiotic resistance*, J. Environ. Econ. Manage., 2001 · Ostrom, *Governing the Commons*, 1990.

### C3 — Partager le coût d'un plateau technique mutualisé (jeux coopératifs)
**Difficulté : 3/5** · Mots-clés : `jeux coopératifs`, `valeur de Shapley`, `nucléole`, `cœur`

Plusieurs hôpitaux achètent ensemble un équipement coûteux (IRM, robot chirurgical) et doivent se partager la facture. Le groupe compare plusieurs règles de partage et vérifie leur stabilité.
- **MVP** : jeu de coût à 4–6 hôpitaux (volumes et coûts paramétrables) → cœur, valeur de Shapley, nucléole, partage proportionnel.
- **Réussite** : vérification que chaque règle est dans le cœur ou non ; propriétés axiomatiques illustrées ; sensibilité à l'arrivée d'un nouvel hôpital.
- **Extensions** : calcul approché de Shapley pour beaucoup de joueurs, coalitions contraintes par la géographie.
- **Faisabilité** : énumération exacte jusqu'à environ 10 joueurs. Plan B : 3 joueurs avec calcul à la main pour valider le code.

**CoursIA** : [GameTheory-15-CooperativeGames](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-15-CooperativeGames-Python.ipynb) · [GameTheory-15f-Shapley-Groupes](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-15f-Shapley-Groupes-Python.ipynb)
**Références** : Peleg & Sudhölter, *Introduction to the Theory of Cooperative Games*, 2007 · Osborne & Rubinstein, *A Course in Game Theory*, ch. 13-14 (bibliothèque du cours).

### C4 — Jeux de sécurité de Stackelberg pour les inspections sanitaires
**Difficulté : 3/5** · Mots-clés : `Stackelberg`, `engagement`, `stratégies mixtes`, `programmation linéaire`

Une autorité sanitaire ne peut pas inspecter tous les établissements (restaurants, laboratoires). Les établissements observent sa politique avant de décider de respecter les règles. Le groupe calcule une politique d'inspection optimale de Stackelberg.
- **MVP** : jeu de sécurité à N cibles (valeurs et pénalités paramétrables) → stratégie du leader par programmation linéaire (DOBSS ou multiple-LP).
- **Réussite** : comparaison avec une inspection uniforme ou proportionnelle au risque, et gain mesuré ; robustesse à l'incertitude sur les gains des établissements.
- **Extensions** : données ouvertes d'inspection (Alim'confiance) pour calibrer le risque, suiveurs à rationalité limitée (quantal response).
- **Faisabilité** : LP de petite taille avec `scipy` ou `PuLP`. Plan B : 3 cibles avec résolution graphique.

**CoursIA** : [GameTheory-09b-Commitment-Stackelberg](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-09b-Commitment-Stackelberg-Python.ipynb) · [GameTheory-09c-Stackelberg-SecurityGame](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-09c-Stackelberg-SecurityGame-Python.ipynb)
**Références** : Tambe, *Security and Game Theory*, Cambridge University Press, 2011 · [Alim'confiance](https://www.data.gouv.fr/fr/datasets/resultats-des-controles-officiels-sanitaires-dispositif-dinformation-alimconfiance/).

### C5 — Persuasion bayésienne : communiquer un risque sanitaire
**Difficulté : 4/5** · Mots-clés : `persuasion bayésienne`, `design d'information`, `concavification`

Une autorité sanitaire dispose d'une information sur un risque (qualité de l'eau, épidémie locale) et choisit ce qu'elle publie pour influencer les comportements. Le groupe calcule le signal optimal et en discute l'éthique.
- **MVP** : modèle de Kamenica-Gentzkow à deux états → signal optimal par concavification, illustré numériquement.
- **Réussite** : gain de l'émetteur vs révélation complète et silence ; extension à plusieurs états résolue par programmation linéaire.
- **Extensions** : récepteurs hétérogènes, perte de crédibilité dans un jeu répété, comparaison avec des récepteurs simulés par LLM.
- **Faisabilité** : sujet théorique exigeant mais calculatoirement léger. Plan B : rester à deux états et deux actions.

**CoursIA** : [GameTheory-17e-Bayesian-Persuasion](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-17e-Bayesian-Persuasion-Python.ipynb) · [GameTheory-11-BayesianGames](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-11-BayesianGames-Python.ipynb)
**Références** : Kamenica & Gentzkow, *Bayesian Persuasion*, American Economic Review, 2011.

### C6 — Antisélection et contrats d'assurance santé
**Difficulté : 3/5** · Mots-clés : `information asymétrique`, `screening`, `Rothschild-Stiglitz`, `simulation de marché`

Un assureur ne connaît pas le risque réel de ses clients ; les plus risqués sont les plus intéressés par une bonne couverture. Le groupe simule ce marché et cherche les menus de contrats qui séparent les profils.
- **MVP** : modèle à deux types de risque → équilibre séparant de Rothschild-Stiglitz, simulation d'une « spirale de la mort » quand le contrat est unique.
- **Réussite** : conditions d'existence de l'équilibre ; effet chiffré d'une obligation d'assurance ou d'une tarification uniforme.
- **Extensions** : continuum de types, apprentissage du risque par l'assureur (données synthétiques), signaux volontaires (bilans de santé).
- **Faisabilité** : modèle analytique et simulation. Plan B : se limiter au cas à deux types.

**CoursIA** : [GameTheory-17b-Asymmetric-Information](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-17b-Asymmetric-Information-Python.ipynb) · [GameTheory-16-MechanismDesign](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-16-MechanismDesign-Python.ipynb)
**Références** : Rothschild & Stiglitz, *Equilibrium in Competitive Insurance Markets*, QJE, 1976 · Einav & Finkelstein, *Selection in Insurance Markets*, JEP, 2011.

---

## D — Machine learning et deep learning sur données de santé

### D1 — Classification d'ECG calibrée et sélective
**Difficulté : 3/5** · Mots-clés : `ECG`, `réseau convolutif 1D`, `calibration`, `abstention`

Un classifieur d'ECG doit savoir quand il ne sait pas. Le groupe entraîne un modèle de détection de pathologies sur ECG 12 dérivations, puis mesure et améliore sa calibration et sa capacité à s'abstenir.
- **MVP** : PTB-XL (PhysioNet, accès ouvert) → CNN 1D sur 5 super-classes diagnostiques, AUC par classe.
- **Réussite** : courbe de fiabilité et ECE avant/après *temperature scaling* ; courbe risque-couverture quand le modèle s'abstient sur les cas incertains.
- **Extensions** : ensembles profonds, MC dropout, généralisation à un autre jeu d'ECG.
- **Faisabilité** : 21 000 ECG, entraînement possible sur CPU en sous-échantillonnant à 100 Hz. Plan B : caractéristiques extraites + gradient boosting.

**CoursIA** : [ML-04-Evaluation-Python](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/ML.Net/ML-04-Evaluation-Python.ipynb) · [DataScienceWithAgents/03-DeepLearning](https://github.com/jsboige/CoursIA/tree/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/03-DeepLearning)
**Références** : Wagner et al., *PTB-XL, a large publicly available electrocardiography dataset*, Scientific Data, 2020 · Guo et al., *On Calibration of Modern Neural Networks*, ICML, 2017.

### D2 — Stadification automatique du sommeil
**Difficulté : 3/5** · Mots-clés : `EEG`, `séries temporelles`, `HMM`, `classification de séquences`

Un spécialiste du sommeil découpe chaque nuit en époques de 30 s classées en stades (éveil, N1, N2, N3, REM). Le groupe automatise ce découpage et exploite la structure temporelle des transitions entre stades.
- **MVP** : Sleep-EDF (PhysioNet) → caractéristiques spectrales par époque + classifieur, puis lissage par modèle de Markov caché.
- **Réussite** : kappa de Cohen et matrice de confusion par stade, gain mesuré du lissage temporel, comparaison avec un classifieur sans contexte.
- **Extensions** : modèle séquentiel profond, hypnogramme avec incertitude, généralisation entre sujets.
- **Faisabilité** : chargement documenté dans `mne`. Plan B : un seul canal EEG et trois classes.

**CoursIA** : [PyMC-14-Sequences](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-14-Sequences.ipynb) · [ML-05-TimeSeries-Python](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/ML.Net/ML-05-TimeSeries-Python.ipynb)
**Références** : [Sleep-EDF](https://physionet.org/content/sleep-edfx/) · [Tutoriel MNE sleep staging](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html).

### D3 — Apprentissage fédéré entre hôpitaux simulés
**Difficulté : 4/5** · Mots-clés : `federated learning`, `non-IID`, `confidentialité différentielle`, `Flower`

Les hôpitaux ne peuvent pas centraliser leurs données. Le groupe simule plusieurs hôpitaux aux populations différentes et compare un modèle fédéré à des modèles locaux et à un modèle centralisé.
- **MVP** : jeu public tabulaire ou d'imagerie (MedMNIST) partitionné en 4–8 « hôpitaux » non-IID → FedAvg avec Flower.
- **Réussite** : performance fédérée vs locale vs centralisée ; effet du degré d'hétérogénéité ; coût de communication.
- **Extensions** : FedProx, confidentialité différentielle (compromis vie privée/utilité), hôpital malveillant.
- **Faisabilité** : simulation sur une seule machine. Plan B : FedAvg codé à la main en NumPy sur régression logistique.

**CoursIA** : [ML-03-Entrainement&AutoML-Python](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/ML.Net/ML-03-Entrainement%26AutoML-Python.ipynb) · [DataScienceWithAgents/02-ML-Cours](https://github.com/jsboige/CoursIA/tree/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/02-ML-Cours)
**Références** : McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS, 2017 · [Flower](https://flower.ai/).

### D4 — Robustesse au changement d'hôpital en imagerie médicale
**Difficulté : 4/5** · Mots-clés : `domain shift`, `généralisation`, `shortcut learning`, `MedMNIST`

Un modèle d'imagerie excellent dans un hôpital s'effondre souvent dans un autre (autre scanner, autre population). Le groupe mesure cette dégradation et évalue des méthodes pour la limiter.
- **MVP** : jeu multi-sources publié pour la généralisation (Camelyon17-WILDS, histopathologie de 5 hôpitaux) → entraînement sur 3 hôpitaux, test sur les autres.
- **Réussite** : écart in-domain/out-of-domain mesuré ; au moins deux méthodes comparées (augmentation, normalisation de coloration, DANN ou équivalent).
- **Extensions** : détection de données hors distribution, cartes de saillance pour repérer les raccourcis.
- **Faisabilité** : version réduite en petite résolution, entraînement court sur GPU modeste ou Colab. **Différenciation** : des projets de détection de cancer ont déjà été rendus ; ici, le verrou est la **généralisation inter-sites**, pas la performance de classification. Plan B : MedMNIST avec décalage artificiel.

**CoursIA** : [DataScienceWithAgents/04-Vision](https://github.com/jsboige/CoursIA/tree/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/04-Vision) · [ML-04b-ModelComparison-Validity-Python](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/ML.Net/ML-04b-ModelComparison-Validity-Python.ipynb)
**Références** : Koh et al., *WILDS: A Benchmark of in-the-Wild Distribution Shifts*, ICML, 2021 · Geirhos et al., *Shortcut learning in deep neural networks*, Nature MI, 2020.

### D5 — Alerte précoce de sepsis en réanimation
**Difficulté : 3/5** · Mots-clés : `séries temporelles irrégulières`, `valeurs manquantes`, `utilité clinique`

Le sepsis doit être détecté tôt, mais les signes vitaux et les analyses arrivent de façon irrégulière et incomplète. Le groupe construit un modèle d'alerte et l'évalue avec une métrique qui récompense l'anticipation.
- **MVP** : PhysioNet/Computing in Cardiology Challenge 2019 (40 000 séjours, accès ouvert) → imputation + caractéristiques glissantes + gradient boosting.
- **Réussite** : score d'utilité officiel du challenge, AUROC/AUPRC, délai d'anticipation et charge d'alertes par patient.
- **Extensions** : modèle séquentiel, calibration, explication locale des alertes, analyse des faux positifs.
- **Faisabilité** : jeu volumineux mais tabulaire. Plan B : sous-échantillon d'un seul hôpital.

**CoursIA** : [ML-05-TimeSeries-Python](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/ML.Net/ML-05-TimeSeries-Python.ipynb) · [ML-09-Anomaly-Detection-Python](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/ML.Net/ML-09-Anomaly-Detection-Python.ipynb)
**Références** : Reyna et al., *Early Prediction of Sepsis From Clinical Data: the PhysioNet/Computing in Cardiology Challenge 2019*, Crit. Care Med., 2020.

---

## E — IA générative appliquée à la santé

La filière n'a pas de cours dédié à l'IA générative : ces sujets en reprennent les outils, toujours couplés à une mesure rigoureuse. Ils nécessitent un accès à un LLM, via un service avec clé ou un modèle local.

### E1 — Du texte libre au réseau bayésien : orientation de symptômes avec abstention
**Difficulté : 3/5** · Mots-clés : `extraction structurée`, `réseau bayésien`, `abstention`, `Synthea`

Un patient décrit ses symptômes en langage naturel. Un LLM extrait des variables structurées, puis un réseau bayésien explicite calcule les probabilités. Le système doit s'abstenir quand l'extraction ou l'inférence est trop incertaine.
- **MVP** : patients synthétiques Synthea → réseau bayésien `pgmpy` sur un périmètre fermé (par ex. infections respiratoires) + extraction JSON par LLM depuis des descriptions générées.
- **Réussite** : précision d'extraction par variable ; comparaison avec un LLM qui répond directement ; taux d'abstention et erreurs évitées.
- **Extensions** : question de clarification choisie par valeur de l'information, explication du raisonnement.
- **Faisabilité** : périmètre volontairement étroit. Plan B : extraction par règles si le LLM n'est pas disponible. Le démonstrateur ne pose aucun diagnostic réel.

**CoursIA** : [PyMC-04-Bayesian-Networks](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-04-Bayesian-Networks.ipynb) · [DecInfer-07-Expert-Systems](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/DecisionTheory/DecInfer/DecInfer-07-Expert-Systems.ipynb) · [GenAI/CaseStudies/Medical-Chatbot](https://github.com/jsboige/CoursIA/tree/main/MyIA.AI.Notebooks/GenAI/CaseStudies/Medical-Chatbot)
**Références** : [Synthea](https://synthetichealth.github.io/synthea/) · [pgmpy](https://pgmpy.org/).

### E2 — Vérification d'affirmations de santé avec citations
**Difficulté : 3/5** · Mots-clés : `fact-checking`, `RAG`, `citations`, `infodémie`

Les affirmations de santé circulent plus vite que leurs démentis. Le groupe construit un système qui classe une affirmation (vraie, fausse, mitigée, non vérifiable) en citant les passages de sources fiables sur lesquels il s'appuie.
- **MVP** : jeu PUBHEALTH (affirmations de santé annotées avec explications) → recherche dans un corpus de sources + classification par LLM avec citation obligatoire.
- **Réussite** : exactitude vs baseline sans recherche ; fidélité des citations vérifiée sur un échantillon ; taux d'abstention.
- **Extensions** : détection d'affirmations dans un texte long, calibration de la confiance, robustesse aux reformulations.
- **Faisabilité** : jeu public de 11 800 affirmations. **Différenciation** : des détecteurs de fake news par classification (BERT) ont déjà été rendus ; ici, le verrou est la **justification par citations vérifiables** et l'abstention. Plan B : petit modèle d'inférence textuelle (NLI) sans génération.

**CoursIA** : [GenAI](https://github.com/jsboige/CoursIA/tree/main/MyIA.AI.Notebooks/GenAI) (séries texte et RAG) · [SW-12-Python-GraphRAG](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/SymbolicAI/SemanticWeb/SW-12-Python-GraphRAG.ipynb)
**Références** : Kotonya & Toni, *Explainable Automated Fact-Checking for Public Health Claims*, EMNLP, 2020.

### E3 — Données de patients synthétiques : fidélité, utilité et confidentialité
**Difficulté : 3/5** · Mots-clés : `données synthétiques`, `vie privée`, `réseau bayésien`, `CTGAN`

Les données synthétiques permettent de partager sans exposer les patients, à condition de mesurer ce qu'elles préservent et ce qu'elles fuient. Le groupe compare plusieurs générateurs sur ces trois axes.
- **MVP** : un jeu tabulaire public (par ex. *Heart Disease* UCI ou *Diabetes 130*) → au moins deux générateurs (réseau bayésien, CTGAN via SDV) évalués en fidélité statistique et en utilité (entraîner sur synthétique, tester sur réel).
- **Réussite** : au moins une mesure de risque (distance au plus proche enregistrement, attaque d'inférence d'appartenance) et un compromis chiffré.
- **Extensions** : générateur à confidentialité différentielle, génération par LLM, sous-groupes rares.
- **Faisabilité** : bibliothèques `sdv` et `synthcity`. **Différenciation** : des GANs ont déjà été présentés ; ici, le verrou est l'**évaluation de la confidentialité** de données tabulaires. Plan B : un seul générateur et l'évaluation complète.

**CoursIA** : [PyMC-04-Bayesian-Networks](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/PyMC/PyMC-04-Bayesian-Networks.ipynb) · [DataScienceWithAgents/02-ML-Cours](https://github.com/jsboige/CoursIA/tree/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/02-ML-Cours)
**Références** : Jordon et al., *Synthetic Data — what, why and how?*, Royal Society/Alan Turing Institute, 2022 · [SDV](https://docs.sdv.dev/).

---

## F — Hors santé (débordement)

### F1 — Pragmatique bayésienne : le modèle RSA
**Difficulté : 4/5** · Mots-clés : `Rational Speech Acts`, `programmation probabiliste`, `Pyro`, `pragmatique`

Comprendre « certains » comme « pas tous », ou une hyperbole comme une exagération, demande de raisonner sur l'intention du locuteur. Le modèle *Rational Speech Acts* formalise ce raisonnement récursif par inférence bayésienne.
- **MVP** : implémentation RSA de base (implicatures scalaires) puis du modèle d'hyperbole, avec des prédictions comparées aux données expérimentales publiées.
- **Réussite** : reproduction quantitative d'au moins un résultat publié ; analyse de sensibilité aux paramètres de rationalité.
- **Extensions** : comparaison avec les interprétations d'un LLM, référence ambiguë dans un jeu de dialogue.
- **Faisabilité** : modèles petits, inférence exacte possible. Plan B : énumération en NumPy sans Pyro.

**CoursIA** : [Pyro_RSA_Hyperbole](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/Probas/Applications/Pyro_RSA_Hyperbole.ipynb)
**Références** : Goodman & Frank, *Pragmatic Language Interpretation as Probabilistic Inference*, Trends Cogn. Sci., 2016 · [problang.org](https://www.problang.org/).

### F2 — Jeux à champ moyen : évacuation d'une foule
**Difficulté : 4/5** · Mots-clés : `mean field games`, `équations couplées`, `foules`

Quand des milliers d'agents se déplacent en anticipant la congestion créée par les autres, on peut remplacer les agents par une densité. Le groupe résout un jeu à champ moyen d'évacuation d'une salle et le compare à une simulation multi-agents.
- **MVP** : MFG 1D ou 2D discrétisé (équation de Hamilton-Jacobi-Bellman rétrograde + Fokker-Planck progressive) résolu par itération de point fixe.
- **Réussite** : convergence vérifiée, comparaison avec des agents myopes, effet du nombre et de la position des sorties.
- **Extensions** : obstacles, populations hétérogènes, apprentissage par renforcement à champ moyen.
- **Faisabilité** : sujet mathématique ; démarrer en 1D. Plan B : jeu de congestion discret sur graphe.

**CoursIA** : [GameTheory-14-DifferentialGames](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-14-DifferentialGames-Python.ipynb) · [GameTheory-17-MultiAgent-RL](https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/GameTheory/GameTheory-17-MultiAgent-RL-Python.ipynb)
**Références** : Lasry & Lions, *Mean field games*, Japanese Journal of Mathematics, 2007 · Achdou & Laurière, *Mean Field Games and Applications: Numerical Aspects*, 2020.
