# fnn_v22_ft_st_cum vs fnn_v21_ft_st  (bench v22_st_cum)

## Heart head, held-out finetuning set

```
    model_version  best_ft_log_loss  best_ft_brier  best_ft_auc  best_ft_avg_precision  best_ft_best_f1  best_ft_ece
    fnn_v21_ft_st           0.05485        0.01296      0.91902                0.46089          0.48365      0.00066
fnn_v22_ft_st_cum           0.05455        0.01290      0.92005                0.46502          0.48811      0.00068
```

## Brier, F1 and recall on the same held-out set (from cached logits)

```
            model   brier     auc  avg_precision  precision@0.5  recall@0.5  f1@0.5  n_pred_pos@0.5  best_f1  precision@bestF1  recall@bestF1  threshold  recall top 5%  recall top 10%
    fnn_v21_ft_st 0.01296 0.91902        0.46089        0.68250     0.29272 0.40972           10167  0.48365           0.48621        0.48112    0.23528        0.67906         0.77941
fnn_v22_ft_st_cum 0.01290 0.92005        0.46502        0.68172     0.29935 0.41602           10409  0.48811           0.48877        0.48745    0.23519        0.68319         0.78144
```

## Does upgrading look like an upgrade?

Model-implied logit change from swapping a plain copy for an upgraded one on held-out decks. Most upgrades help, so fewer negative effects (especially for rare states) means upgrade states are being learned better.

```
                                       cards   n  mean effect fnn_v21_ft_st  mean effect fnn_v22_ft_st_cum  share <0 fnn_v21_ft_st  share <0 fnn_v22_ft_st_cum  corr(effects)
                                   all cards 344                      0.157                          0.123                   0.244                       0.282          0.791
rare state (<5000 rows in plain or upgraded)  90                      0.065                          0.024                   0.389                       0.422          0.397
                              well supported 254                      0.190                          0.159                   0.193                       0.232          0.847
```

Lowest upgrade effects under fnn_v22_ft_st_cum:

```
         card       char  n_train_plain  n_train_upgraded  upgrade_effect_fnn_v21_ft_st  upgrade_effect_fnn_v22_ft_st_cum
lessonlearned    watcher          20825              8806                        -0.583                            -0.359
   battlehymn    watcher          28747             14386                        -0.442                            -0.312
      glacier     defect         124791             18854                        -0.322                            -0.278
       unload the_silent          19827              5109                         0.086                            -0.235
  autoshields     defect          39376              5671                        -0.031                            -0.229
       scrape     defect          18124              4272                         0.168                            -0.227
 perseverance    watcher          14870              5744                        -0.019                            -0.222
         bash   ironclad        1501563            377583                        -0.239                            -0.209
```

Biggest changes fnn_v21_ft_st to fnn_v22_ft_st_cum:

```
             card       char  n_train_plain  n_train_upgraded  upgrade_effect_fnn_v21_ft_st  upgrade_effect_fnn_v22_ft_st_cum
      outmaneuver the_silent          18850              4934                         0.583                             0.085
           melter     defect          60024              8600                         0.508                             0.035
     darkshackles   ironclad           8240              1220                        -0.487                            -0.036
    wreathofflame    watcher           5456              2086                         0.255                            -0.189
            setup the_silent           7600              3728                         0.363                            -0.074
          rebound     defect          57530              6492                         0.510                             0.081
            slice the_silent          67007              7131                         0.446                             0.019
underhandedstrike the_silent          37915              6439                         0.238                            -0.168
```

## Held-out calibration of each card's plain / upgraded copy, by training support

Lower mean z2 and rms error is better calibrated. The gain from sharing should show in the low-support rows.

```
train rows of state  cells  mean z2 fnn_v21_ft_st  rms rel err fnn_v21_ft_st  mean z2 fnn_v22_ft_st_cum  rms rel err fnn_v22_ft_st_cum
             0-1000      7                  1.423                      0.276                      1.437                          0.259
          1000-5000     80                  1.441                      0.169                      1.451                          0.177
         5000-20000    244                  1.255                      0.095                      1.278                          0.099
          20000-inf    333                  1.117                      0.055                      1.193                          0.055
```

## Upgrade x combo: is a pair's interaction the same in every upgrade state?

Spread is max minus min of the pair interaction over the four plain/upgraded combinations. rarest/best near 1 means what was learned in the well-supported states carries over to the rare ones.

```
                           pair set  pairs  mean spread fnn_v21_ft_st  mean spread fnn_v22_ft_st_cum  strong pairs  rarest/best fnn_v21_ft_st  rarest/best fnn_v22_ft_st_cum
                          all pairs    190                      0.018                          0.019             9                      0.347                          0.663
rarest state combination <5000 rows    187                      0.018                          0.018             9                      0.347                          0.663
                  rarest <1000 rows    119                      0.015                          0.016             3                      0.485                          1.303
```

Pairs whose spread changed most:

```
                       combo  n_best  n_rarest  spread fnn_v21_ft_st  spread fnn_v22_ft_st_cum  best fnn_v22_ft_st_cum  rarest fnn_v22_ft_st_cum
defragment + biasedcognition    6865      1602                 0.018                     0.061                  -0.036                    -0.036
bouncingflask + piercingwail   10210      1428                 0.021                     0.058                   0.099                     0.066
      heavyblade + demonform    5097      2988                 0.025                     0.057                   0.087                     0.041
    catalyst + bouncingflask    8000      3615                 0.044                     0.011                   0.015                     0.018
    corruption + darkembrace    3772      1161                 0.016                     0.047                  -0.013                     0.003
         exhume + corruption    2082       336                 0.039                     0.007                   0.024                     0.031
 afterimage + infiniteblades    3878       981                 0.008                     0.039                   0.019                     0.058
  alloutattack + wraithformv    1147       519                 0.010                     0.039                   0.024                     0.038
```

## Combos: model interaction and what it leaves unexplained (logit)

Residual > 0 means the model undervalues the combo. Compare against the null controls for the noise level.

```
                               set   n  interaction fnn_v21_ft_st  resid train fnn_v21_ft_st  resid test fnn_v21_ft_st  interaction fnn_v22_ft_st_cum  resid train fnn_v22_ft_st_cum  resid test fnn_v22_ft_st_cum
                     listed combos 111                      0.001                      0.006                     0.063                          0.002                          0.003                         0.067
listed pairs with upgradable cards  90                     -0.000                     -0.017                     0.019                          0.001                         -0.021                         0.021
         listed, relics or triples  21                      0.005                      0.105                     0.252                          0.004                          0.104                         0.262
                     null controls 100                      0.002                      0.002                     0.016                          0.003                         -0.001                         0.013
```

Headline combos:

```
                          combo  n_train  n_test  model_interaction_fnn_v21_ft_st  residual_train_fnn_v21_ft_st  residual_test_fnn_v21_ft_st  model_interaction_fnn_v22_ft_st_cum  residual_train_fnn_v22_ft_st_cum  residual_test_fnn_v22_ft_st_cum
           barricade + entrench     9331    2152                            0.025                         0.457                        0.341                                0.025                             0.437                            0.314
barricade + entrench + bodyslam     5315    1196                            0.001                         0.055                        0.724                                0.002                             0.066                            0.763
        corruption + feelnopain    12789    3123                            0.028                         0.095                       -0.041                                0.023                             0.072                           -0.067
       corruption + darkembrace     8573    1995                            0.008                         0.019                        0.112                               -0.011                             0.047                            0.136
        corruption + deadbranch     4043     964                            0.022                         0.895                        1.083                                0.070                             0.836                            0.997
```

## Combo summary

```
                     metric             0                 1
                      model fnn_v21_ft_st fnn_v22_ft_st_cum
                  synergy_n           111               111
  synergy_model_interaction         0.001             0.002
     synergy_residual_train         0.006             0.003
 synergy_abs_residual_train         0.132             0.130
  synergy_share_z_gt2_train         0.072             0.054
synergy_share_z_lt_m2_train         0.090             0.090
      synergy_residual_test         0.063             0.067
  synergy_abs_residual_test         0.235             0.230
   synergy_share_z_gt2_test         0.072             0.063
 synergy_share_z_lt_m2_test         0.018             0.018
                     null_n           100               100
     null_model_interaction         0.002             0.003
        null_residual_train         0.002            -0.001
    null_abs_residual_train         0.104             0.097
     null_share_z_gt2_train         0.030             0.010
   null_share_z_lt_m2_train         0.040             0.040
         null_residual_test         0.016             0.013
     null_abs_residual_test         0.228             0.228
      null_share_z_gt2_test         0.040             0.030
    null_share_z_lt_m2_test         0.030             0.040
```

## Card picks

```
            model  agree_all  agree_heart_killers  agree_others  picks_a_card  decisions  cards_calibrated  mean_z2  share_abs_z_gt2  rms_rel_err
    fnn_v21_ft_st      0.402                0.552         0.398         0.707 509451.000           173.000    1.129            0.064        0.176
fnn_v22_ft_st_cum      0.402                0.556         0.398         0.734 509451.000           173.000    1.123            0.069        0.178
```
