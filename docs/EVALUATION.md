# Modalités d'évaluation

## Taille des groupes

Le projet est conçu pour un **groupe de 3 personnes**.

| Taille du groupe | Ajustement appliqué à la note finale |
|---:|---:|
| 1 personne | +3 points |
| 2 personnes | +1 point |
| 3 personnes | aucun ajustement |
| 4 personnes | −1 point |

Un groupe de plus de quatre personnes n'est pas autorisé sans accord explicite de l'enseignant.

## Soutenance collégiale

La soutenance a lieu le **mercredi 13 janvier 2027, de 08:30 à 12:15, salle A203 C**, lors de la dernière séance du module. Les pairs et l'enseignant évaluent chaque projet sur les mêmes quatre critères, chacun noté de 0 à 10.

| Critère | Ce qui est évalué |
|---|---|
| Présentation | Clarté, structure, respect du temps, qualité de la démonstration et réponses aux questions |
| Théorie | Compréhension des méthodes, contexte médical ou applicatif, justification des choix, limites et travaux cités |
| Technique | Fonctionnement, qualité du code, tests, mesures, performances et reproductibilité |
| Organisation | Répartition du travail, historique Git, documentation, gestion des risques et progression |

Pour un jury donné, la **note de base sur 20** est :

```text
(présentation + théorie + technique + organisation) / 2
```

La moyenne des évaluations des pairs forme 50 % de la note de soutenance et l'évaluation de l'enseignant forme les 50 % restants. L'évaluation par les pairs se fait par un formulaire communiqué le jour de la soutenance. Les évaluations manifestement incomplètes ou non argumentées peuvent être écartées.

```text
note_soutenance = 50 % moyenne des pairs + 50 % évaluation de l'enseignant
note_finale = min(20, note_soutenance + ajustement de taille du groupe)
```

## Attendus transverses

La difficulté affichée dans le catalogue indique l'ampleur technique du sujet, mais ne remplace pas les critères ci-dessus. Une excellente réalisation d'un sujet 2/5 peut obtenir une excellente note. Sont particulièrement valorisés :

- un MVP réellement fonctionnel ;
- des résultats mesurés plutôt que des impressions ;
- la comparaison à une baseline simple ;
- une quantification honnête de l'incertitude ;
- une analyse des échecs et des limites, y compris du point de vue clinique ;
- des tests reproductibles ;
- une contribution visible de chaque membre.

## Remise

La Pull Request finale doit être ouverte au plus tard le **lundi 11 janvier 2027 à 23:59**, soit deux jours avant la soutenance. Le contenu évalué est celui de la PR à cette échéance. Les corrections ultérieures doivent être clairement identifiées.
