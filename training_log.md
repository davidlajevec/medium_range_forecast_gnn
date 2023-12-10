# TRIED ALREADY

1) različno število local custom layerjev brez skip connectionv
2) skip connections na zadnji plasti
3) skip connections na vsaki plasti, 5 in 12 plasti


# WORKS BEST

1) skip connections na vsak plasti 8 plasti, 128 hidden channels


# TO DO LGCN

1) NN agregation

2) testirat k1 in k2 povezave

3) različni scatter reduce to try: amax, amin, sum, prod ....

4) leaky_relu, elu -- RUNNING

5) testirat razične načine kako dodati node feature v agregaciji/updatu

6) mae za loss funkcijo??

7) multihead attention pri agregacija

8) dropout, batch norm

9) DODAT PODATKE

