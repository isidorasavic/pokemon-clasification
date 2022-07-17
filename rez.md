--------------- bez obelezja tipa liste -------------------------
### HistGradientBoostingClassifier
 1. learning-rate = 0.09, bez imena i bez popunjavanja NaN polja
    * 0.3834
 2. learning-rate = 0.09, bez imena i sa popunjavanjem NaN polja (za pol, i rate hvatanja i bezanja)
    * 0.33678
 3. learning-rate = 0.09, bez imena i sa popunjavanjem NaN polja (rate hvatanja i bezanja)
    * 0.36269
 4. lr = 0.09 i umesto dva polja za intervale, samo sirina intervala
    * 0.36787
 5. sve isto samo categorizacija celih weakness i sirina inervala (isto i za cele intervale)
    * 0.77
 6. lr = 0.09 samo i resistance i weakness celi kao broj i sirina inervala (isto i za cele intervale)
    * 0.839
 7. isto kao 6. samo popunjena prazna polja za pol i rate
    * 0.829
 8. sa clasifikacijom lista moci na 6. korak "poss_attacks"
    * 0.860

### BaggingClassifier
 1. Popunjeni polovi i rate
    * 0.31088

### KNN (popunjno sve Nan)
 1. k = 30
    * 0.2124
 2. k = 25
    * 0.22279
 
### BaggingClassifier (popunjeno NaN)
 1. 500 estimatora i random_state 8
    * 0.3212
 2. 500 estimatora i random_state 10
    * 0.326
 3. 500 estimatora i random_state 15
     * 0.3367 
