# Données

Le projet utilise un jeu **synthétique** de 1 200 annonces, généré par
`senegal_rental_price.data.generate` avec une graine fixe (42).

Cette option a été retenue car elle est reproductible, ne dépend d'aucun site tiers et ne
présente aucun risque juridique lié au scraping. Les prix sont construits à partir de règles
explicites : Dakar est plus chère, certains quartiers dakarois ont une prime, puis la surface,
le type de bien, le caractère meublé et les équipements modifient le prix. Un bruit
log-normal simule la variabilité du marché.

Ces données ne sont pas des observations du marché réel. Le modèle sert à démontrer une
chaîne MLOps et ne doit pas être utilisé pour fixer un vrai loyer.

- `raw/` : génération initiale, non versionnée ;
- `processed/` : version nettoyée utilisée pour l'entraînement.

