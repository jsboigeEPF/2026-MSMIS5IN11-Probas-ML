# 2026 - EPF MSMIS5IN11 - IA probabiliste, théorie des jeux et machine learning

Dépôt de projets de la **seconde partie** du cours **MSMIS5IN11 — Intelligence Artificielle : Algorithmes d'Apprentissage**, majeure **Ingénierie et santé**, EPF, année 2026-2027. Les projets explorent l'IA probabiliste, la théorie des jeux et le machine learning, appliqués de préférence à la santé.

La première partie du cours (IA exploratoire et symbolique) a son propre dépôt : [jsboigeEPF/2026-MSMIS5IN11-Search-Symbolic](https://github.com/jsboigeEPF/2026-MSMIS5IN11-Search-Symbolic).

## Calendrier

| Étape | Date |
|---|---|
| Présentation des sujets | Mercredi 9 décembre 2026, première séance du chapitre 2 |
| Dépôt de la Pull Request finale | **Lundi 11 janvier 2027 à 23:59** |
| Soutenance et démonstration | **Mercredi 13 janvier 2027, 08:30–12:15**, salle A203 C (dernière séance du module) |

Séances du chapitre 2 : mercredi 9 décembre 2026 (matin et après-midi), mardi 12 janvier 2027 (après-midi), mercredi 13 janvier 2027 (matin).

## Démarrage rapide

1. Constituez un groupe — **3 personnes est le format standard**.
2. Choisissez un sujet dans le [catalogue 2026-2027](docs/CATALOGUE.md), puis [réservez-le par une issue](https://github.com/jsboigeEPF/2026-MSMIS5IN11-Probas-ML/issues/new?template=inscription-groupe.yml). Un sujet ne peut être réservé que par un groupe. Reportez aussi votre groupe et votre sujet dans la feuille d'inscription indiquée sur Moodle.
3. Forkez ce dépôt sur le compte d'un membre du groupe.
4. Créez un dossier unique à la racine, nommé `groupe-<nom-court>`.
5. Copiez le contenu de [`template-groupe/`](template-groupe/) dans ce dossier.
6. Travaillez par commits réguliers sans modifier les fichiers des autres groupes.
7. Si le groupe est bloqué, ouvrez tôt une [demande d'accompagnement](https://github.com/jsboigeEPF/2026-MSMIS5IN11-Probas-ML/issues/new?template=demande-accompagnement.yml).
8. Ouvrez une Pull Request vers ce dépôt avant la date limite.

Les sujets libres sont possibles, mais doivent être proposés avec le [formulaire dédié](https://github.com/jsboigeEPF/2026-MSMIS5IN11-Probas-ML/issues/new?template=proposition-sujet.yml) et validés avant développement.

## Documents de référence

- [Catalogue des sujets 2026-2027](docs/CATALOGUE.md)
- [Guide de réalisation et de soumission](docs/GUIDE-SOUMISSION.md)
- [Modalités d'évaluation](docs/EVALUATION.md)
- [Sujets réalisés lors des promotions précédentes](docs/HISTORIQUE-SUJETS.md)
- [Gabarit de dossier de groupe](template-groupe/README.md)
- [Charte pour les agents de code](AGENTS.md)

## Catégories du catalogue

| Code | Catégorie |
|---|---|
| A | Inférence bayésienne et programmation probabiliste |
| B | Décision sous incertitude et causalité |
| C | Théorie des jeux et mécanismes en santé |
| D | Machine learning et deep learning sur données de santé |
| E | IA générative appliquée à la santé |
| F | Hors santé (débordement) |

## Livrables attendus

Chaque groupe fournit au minimum :

- le code source et des instructions reproductibles ;
- un `README.md` complet ;
- un notebook explicatif avec analyses et visualisations **ou** une interface/démonstration fonctionnelle ;
- des tests adaptés au projet ;
- les slides de soutenance ;
- un fichier `.env.example` sans valeur secrète si le projet utilise des services externes ;
- un historique Git montrant la participation des membres.

Les jeux de données téléchargés, poids de modèles, environnements virtuels, fichiers `.env` et clés API ne doivent jamais être commités.

## Ressources du cours

- **Notebooks** : le dépôt public [CoursIA](https://github.com/jsboige/CoursIA), en particulier les séries [Probas](https://github.com/jsboige/CoursIA/tree/main/MyIA.AI.Notebooks/Probas) (Infer.NET, PyMC, théorie de la décision, causalité), [GameTheory](https://github.com/jsboige/CoursIA/tree/main/MyIA.AI.Notebooks/GameTheory) et [ML](https://github.com/jsboige/CoursIA/tree/main/MyIA.AI.Notebooks/ML). Les fiches du catalogue y renvoient. Un projet ne se limite pas à recopier un notebook : il construit un démonstrateur, mesure ses résultats et documente ses choix.
- **Bibliographie** : la bibliothèque du cours (lien en lecture seule sur Moodle). Les ouvrages ne doivent pas être recopiés dans le dépôt ; citez-les.

## Licence

Le contenu pédagogique de ce dépôt est distribué sous [licence MIT](LICENSE). Chaque groupe reste responsable des licences des modèles, données, bibliothèques et médias qu'il utilise.
