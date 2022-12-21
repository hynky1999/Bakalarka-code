We started with this:
[('irozhlas.json', 200554), ('denik.json', 1006447), ('aktualne.json', 163792), ('idnes.json', 507828), ('novinky.json', 443767), ('ihned.json', 33672), ('seznamzpravy.json', 75204)]


# Filterig by czech language

## We tried to filter by czech language with zero tolerance results are as follows.

[('irozhlas.json', 180372), ('denik.json', 836460), ('aktualne.json', 120186), ('idnes.json', 303369), ('novinky.json', 360855), ('ihned.json', 24537), ('seznamzpravy.json', 67922)]
false


We briefly inspected each files and found of the following:
Ihned.cz:
All of the observed filtered articles were in czech language. The reason why they were filtered was beacause
at the end there were sport results. Or there was a sport match with live results and comentary. 

Example:
Anglická fotbalová liga - 24. kolo:
Liverpool - Leicester 1:1 (3. Mané - 45.+2 Maguire), Tottenham - Watford 2:1 (80. Son Hung-min, 87. Llorente - 38. Cathcart), Bournemouth - Chelsea 4:0 (47. a 74. King, 63. Brooks, 90.+5 Daniels), Southampton - Crystal Palace 1:1 (77. Ward-Prowse - 41. Zaha).
Ve Francii vzpomínali na Sala
Fotbalisté Nantes vzdali hold svému bývalému spoluhráči Emilianu Salovi, který zmizel při letu do nového působiště v Cardiffu. První duel ve francouzské lize, k němuž po zmizení argentinského útočníka tým nastoupil, provázely od počátku velké emoce.

Aktualne.cz:
Had similiaar problem with sports but there were also some english articles finally. However they never had more than 0.2 confidence.


Idnes.cz:
While we haven't found any english articles, we found a lot of articles
with tables which. Which made it obviously hard to to parse as czech

Example:
Normativní náklady se liší podle toho, zda jste nájemce nebo vlastník bytu. U nájemních bytů došlo k navýšení o 15-27 procent, u družstevních a vlastních bytů o 7-8 procent.  Do nákladů na bydlení se započítává nájemné, výdaje za energii, vodné, stočné, odpady a vytápění.
Normativní náklady na bydlení pro nájemní byty platné od 1.1.2010 do 31.1.2010 (v Kč)
Počet osob v rodině
 Počet obyvatel obce
Praha
nad 100 000 
50 000-99 999 
10 000-49 999
do 9 999 
 1
5 877
4 816
4 597
4 309
4 016
 2
8 489


Seznamzpravy.cz:
Same as idnes.cz mostly lists and tables. No english articles.


We also decided not to filter by brief as it was not reliable.

Irozhlas.cz
few english articles  same as others.


Novinky.cz
no english

idnes.cz very few english