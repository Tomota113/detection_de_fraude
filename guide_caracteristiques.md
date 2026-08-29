# 📊 Guide d'Explication des Caractéristiques (Features) de Détection de Fraude

Ce guide explique de manière très simple **chaque indicateur** (caractéristique ou "feature") utilisé par l'Intelligence Artificielle pour analyser la fraude. Pour chaque indicateur, vous trouverez une explication simple, un exemple de client honnête et un exemple de client fraudeur.

---

## 1. Les Indicateurs de Consommation brute

Ces indicateurs étudient les volumes d'électricité consommés par le client.

### 📉 `conso_mean` (Consommation Moyenne)
* **Qu'est-ce que c'est ?** La quantité moyenne d'électricité (en kWh) consommée par le client par facture.
* **Exemple Client Normal :** Consomme en moyenne 250 kWh par mois (un foyer classique).
* **Exemple Client Suspect :** Consomme en moyenne 12 kWh par mois. Pour une maison entière, c'est tellement bas que cela suggère que le compteur a été ralenti ou ponté (l'électricité passe à côté du compteur).

### 🔀 `conso_coeff_variation` (Coefficient de Variation) ⚡ *TRÈS PUISSANT*
* **Qu'est-ce que c'est ?** C'est un indicateur de la **stabilité** de la consommation. Plus le chiffre est grand, plus la consommation fait les "montagnes russes".
* **Exemple Client Normal (CV faible, ex: 0.10 soit 10%) :**
  - Janvier : 150 kWh | Février : 160 kWh | Mars : 145 kWh. La consommation est stable.
* **Exemple Client Suspect (CV élevé, ex: 1.30 soit 130%) :**
  - Janvier : 400 kWh | Février : 0 kWh | Mars : 5 kWh | Avril : 390 kWh. 
  - **Pourquoi c'est suspect ?** Cette instabilité extrême trahit un client qui "bloque" son compteur à sa guise (ex: pendant qu'il utilise de gros appareils) puis le remet en marche avant que le releveur ne vienne.

---

## 2. Les Indicateurs Temporels (Le comportement dans le temps)

Ces indicateurs analysent l'évolution des habitudes du client sur plusieurs années.

### 📉 `tendance_conso` (Pente de Consommation) ⚡ *TRÈS PUISSANT*
* **Qu'est-ce que c'est ?** L'évolution générale de la consommation. Est-ce qu'elle monte, reste stable, ou descend au fil des ans ?
* **Exemple Client Normal (Pente = 0.02) :** Sa consommation augmente légèrement chaque année au fur et à mesure qu'il achète de nouveaux appareils électriques.
* **Exemple Client Suspect (Pente = -0.15) :** Sa courbe de consommation plonge vers le bas de manière continue depuis 5 ans, alors que le climat ou la taille de sa maison n'ont pas changé. C'est l'indice d'une fraude progressive.

### 📉 `ratio_derniere_moyenne` (Chute Récente)
* **Qu'est-ce que c'est ?** Compare la toute dernière facture reçue avec la moyenne de tout l'historique du client.
* **Exemple Client Normal (Ratio = 0.95) :** Sa dernière facture (235 kWh) est très proche de sa moyenne habituelle (250 kWh).
* **Exemple Client Suspect (Ratio = 0.15) :** Sa dernière facture indique 30 kWh alors que sa moyenne historique est de 300 kWh. Sa consommation a soudainement chuté de **85%**. S'il n'y a pas eu déménagement, c'est le signe d'une fraude qui vient de commencer.

### ❄️☀️ `saison_variation` (Variation Saisonnière)
* **Qu'est-ce que c'est ?** Mesure la différence entre la saison où le client consomme le plus et celle où il consomme le moins.
* **Exemple Client Normal (Variation = 1.3) :** Consomme un peu plus en été pour la climatisation (300 kWh) qu'en hiver (230 kWh).
* **Exemple Client Suspect (Variation = 3.5) :** Consomme 450 kWh en été et passe mystérieusement à 10 kWh en hiver. Une telle variation est disproportionnée.

---

## 3. Les Clignotants Spécifiques à la Fraude

Ces indicateurs ont été créés sur mesure car ils ciblent des comportements typiques de fraudeurs.

### 🎯 `max_drop_ratio` (Chute Brutale Maximale)
* **Qu'est-ce que c'est ?** La plus forte baisse de consommation enregistrée d'une facture à la suivante.
* **Exemple Client Normal :** Baisse maximale de 20% au printemps.
* **Exemple Client Suspect (max_drop = 0.90) :** D'un mois sur l'autre, sa facture est passée de 400 kWh à 40 kWh (une chute brutale de **90%**). C'est le signal d'un compteur qui a été désactivé ou ponté d'un coup.

### 🏷️ `ratio_level1_total` (Le Piège du Palier 1) ⚡ *TRÈS PUISSANT*
* **Qu'est-ce que c'est ?** Les tarifs de la STEG fonctionnent par paliers : les premiers kWh (Palier 1) sont très bon marché, puis les prix augmentent fortement au Palier 2, 3, etc. 
* **Exemple Client Normal (Ratio = 0.40) :** Sa consommation se répartit normalement sur les différents paliers selon ses besoins.
* **Exemple Client Suspect (Ratio = 1.00 soit 100%) :** Toutes ses factures, sans exception, s'arrêtent pile à la limite du Palier 1 (par exemple, 100 kWh).
* **Pourquoi c'est suspect ?** Il est statistiquement impossible qu'un foyer consomme exactement la même quantité minimale chaque mois. Cela prouve que le client bloque son compteur dès qu'il atteint le plafond du tarif pas cher pour ne jamais payer le tarif supérieur.

### 📭 `ratio_zero_conso` (Factures à Zéro)
* **Qu'est-ce que c'est ?** Le pourcentage de factures dans l'historique qui affichent une consommation de **0 kWh**.
* **Exemple Client Normal :** 0% de factures à zéro (sauf cas exceptionnel de vacances prolongées).
* **Exemple Client Suspect (Ratio = 0.40) :** 40% de ses factures indiquent 0 kWh de consommation, alors que le contrat est toujours actif.

---

## 4. Les Indicateurs Liés au Compteur (Matériel)

Ces données proviennent de l'appareil physique et des notes du technicien.

### ⚙️ `has_counter_anomaly` (Anomalie Compteur)
* **Qu'est-ce que c'est ?** Si le releveur de la STEG a enregistré un code d'anomalie physique sur le compteur (ex: plomb cassé, cadran cassé, compteur à l'envers).
* **Exemple Client Normal :** Code 0 (Tout est normal).
* **Exemple Client Suspect :** Code supérieur à 0. C'est la preuve matérielle que le compteur a été manipulé.

### 🔄 `nb_compteurs_uniques` (Changements de Compteurs)
* **Qu'est-ce que c'est ?** Le nombre de fois que le compteur physique a été remplacé pour ce client.
* **Exemple Client Normal :** 1 seul compteur en 10 ans.
* **Exemple Client Suspect :** 4 compteurs différents en 5 ans. 
* **Pourquoi c'est suspect ?** Les fraudeurs abîment parfois volontairement leur compteur pour forcer son remplacement et effacer les preuves de manipulation.

---

## 5. Les Indicateurs Géographiques et Administratifs

### 📍 `region` et `disrict`
* **Qu'est-ce que c'est ?** La localisation du client.
* **Exemple :** L'algorithme sait que le district de Tunis Nord ou de Sfax a, historiquement, des taux de fraude différents. Il utilise cela pour ajuster le score de risque global du client.

### ⏳ `client_anciennete_jours`
* **Qu'est-ce que c'est ?** Le nombre de jours depuis l'ouverture du contrat.
* **Pourquoi c'est utile ?** Un client très ancien (ex: 15 ans) offre un historique long qui permet à l'IA d'être beaucoup plus sûre d'elle. À l'inverse, un nouveau client avec une baisse brutale de consommation est plus difficile à juger (il a peut-être juste moins d'appareils).

---

## 💡 Résumé Visuel : Comment l'IA prend sa décision ?

L'IA ne regarde jamais un seul indicateur. Elle les combine. 

```
[ Client avec Tendance Conso qui baisse ] 
                  + 
[ Volatilité (yoyo) très forte ] 
                  + 
[ Ratio Palier 1 proche de 100% ]
                  =
🏆 DÉCISION DE L'IA : Risque de Fraude à 95% -> INSPECTION IMMÉDIATE
```
