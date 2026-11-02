# fnn_v22_beta_act3_mt_cum vs fnn_v21_beta_act3_mt  (bench v22_pretrained)

## Heart head, held-out finetuning set

```
           model_version  best_ft_log_loss  best_ft_auc  best_ft_ece  best_ft_brier  best_ft_avg_precision  best_ft_best_f1
    fnn_v21_beta_act3_mt           0.05389      0.92344      0.00022        0.01281                0.46961          0.49247
fnn_v22_beta_act3_mt_cum           0.05354      0.92446      0.00019        0.01273                0.47466          0.49791
```

## Standard metrics on the held-out set (log loss, AUC, calibration error, Brier, precision, recall, F1 at 0.5)

```
                   model  log_loss     auc     ece   brier  avg_precision  precision@0.5  recall@0.5  f1@0.5  n_pred_pos@0.5  best_f1  precision@bestF1  recall@bestF1  threshold  recall top 5%  recall top 10%
    fnn_v21_beta_act3_mt   0.05389 0.92344 0.00022 0.01281        0.46961        0.68967     0.29213 0.41042           10041  0.49247           0.49426        0.49070    0.23188        0.68922         0.78739
fnn_v22_beta_act3_mt_cum   0.05354 0.92446 0.00019 0.01273        0.47466        0.69036     0.29741 0.41572           10212  0.49791           0.50897        0.48732    0.24414        0.69045         0.79101
```

## Does upgrading look like an upgrade?

Model-implied logit change from swapping a plain copy for an upgraded one on held-out decks. Most upgrades help, so fewer negative effects (especially for rare states) means upgrade states are being learned better.

```
                                       cards   n  mean effect fnn_v21_beta_act3_mt  mean effect fnn_v22_beta_act3_mt_cum  share <0 fnn_v21_beta_act3_mt  share <0 fnn_v22_beta_act3_mt_cum  corr(effects)
                                   all cards 344                             0.231                                 0.202                          0.122                              0.119          0.818
rare state (<5000 rows in plain or upgraded)  90                             0.138                                 0.085                          0.256                              0.233          0.584
                              well supported 254                             0.263                                 0.243                          0.075                              0.079          0.854
```

Lowest upgrade effects under fnn_v22_beta_act3_mt_cum:

```
         card       char  n_train_plain  n_train_upgraded  upgrade_effect_fnn_v21_beta_act3_mt  upgrade_effect_fnn_v22_beta_act3_mt_cum
lessonlearned    watcher          20825              8806                               -0.820                                   -0.546
metamorphosis   ironclad           6903              4961                               -0.024                                   -0.172
    reprogram     defect           5789              1308                                0.216                                   -0.157
         gash     defect         120857             14687                                0.364                                   -0.145
       unload the_silent          19827              5109                                0.073                                   -0.142
       reaper   ironclad          88886             10965                               -0.140                                   -0.129
       strike   ironclad        4854417            466241                               -0.212                                   -0.114
  forethought   ironclad           1886               613                                0.139                                   -0.110
```

Biggest changes fnn_v21_beta_act3_mt to fnn_v22_beta_act3_mt_cum:

```
            card       char  n_train_plain  n_train_upgraded  upgrade_effect_fnn_v21_beta_act3_mt  upgrade_effect_fnn_v22_beta_act3_mt_cum
           setup the_silent           7600              3728                                0.542                                    0.022
            gash     defect         120857             14687                                0.364                                   -0.145
   pathtovictory    watcher          17253              7451                                0.421                                    0.016
        accuracy the_silent          37480             15816                                0.632                                    0.249
       reprogram     defect           5789              1308                                0.216                                   -0.157
   wreathofflame    watcher           5456              2086                                0.462                                    0.116
masterofstrategy    watcher          13146              1915                               -0.159                                    0.180
       aggregate     defect           5367              1410                                0.396                                    0.066
```

## Held-out calibration of each card's plain / upgraded copy, by training support

Lower mean z2 and rms error is better calibrated. The gain from sharing should show in the low-support rows.

```
train rows of state  cells  mean z2 fnn_v21_beta_act3_mt  rms rel err fnn_v21_beta_act3_mt  mean z2 fnn_v22_beta_act3_mt_cum  rms rel err fnn_v22_beta_act3_mt_cum
             0-1000      7                         1.698                             0.310                             1.395                                 0.286
          1000-5000     80                         1.624                             0.181                             1.588                                 0.182
         5000-20000    244                         1.608                             0.112                             1.528                                 0.110
          20000-inf    333                         1.364                             0.061                             1.471                                 0.063
```

## Upgrade x combo: is a pair's interaction the same in every upgrade state?

Spread is max minus min of the pair interaction over the four plain/upgraded combinations. rarest/best near 1 means what was learned in the well-supported states carries over to the rare ones.

```
                           pair set  pairs  mean spread fnn_v21_beta_act3_mt  mean spread fnn_v22_beta_act3_mt_cum  strong pairs  rarest/best fnn_v21_beta_act3_mt  rarest/best fnn_v22_beta_act3_mt_cum
                          all pairs    190                             0.034                                 0.039            68                             0.657                                 1.163
rarest state combination <5000 rows    187                             0.034                                 0.038            67                             0.654                                 1.166
                  rarest <1000 rows    119                             0.032                                 0.032            37                             0.493                                 1.229
```

Pairs whose spread changed most:

```
                       combo  n_best  n_rarest  spread fnn_v21_beta_act3_mt  spread fnn_v22_beta_act3_mt_cum  best fnn_v22_beta_act3_mt_cum  rarest fnn_v22_beta_act3_mt_cum
        barricade + entrench    3439      1605                        0.053                            0.195                          0.348                            0.240
       accuracy + bladedance    3285      2509                        0.013                            0.137                          0.100                            0.149
defragment + biasedcognition    6865      1602                        0.036                            0.149                         -0.213                           -0.194
        barricade + bodyslam    6038      3120                        0.050                            0.150                          0.265                            0.187
   accuracy + cloakanddagger    3420      2448                        0.040                            0.135                          0.206                            0.124
  barricade + swordboomerang    2314       876                        0.011                            0.094                         -0.145                           -0.239
    catalyst + welllaidplans    4635      1855                        0.027                            0.108                         -0.100                           -0.129
     metallicize + barricade    3203      1385                        0.037                            0.114                          0.077                            0.134
```

## Combos: model interaction and what it leaves unexplained (logit)

Residual > 0 means the model undervalues the combo. Compare against the null controls for the noise level.

```
                               set   n  interaction fnn_v21_beta_act3_mt  resid train fnn_v21_beta_act3_mt  resid test fnn_v21_beta_act3_mt  interaction fnn_v22_beta_act3_mt_cum  resid train fnn_v22_beta_act3_mt_cum  resid test fnn_v22_beta_act3_mt_cum
                     listed combos 111                             0.008                            -0.021                            0.034                                 0.011                                -0.033                                0.029
listed pairs with upgradable cards  90                            -0.001                            -0.040                           -0.007                                 0.001                                -0.051                               -0.011
         listed, relics or triples  21                             0.050                             0.061                            0.210                                 0.055                                 0.045                                0.203
                     null controls 100                            -0.000                            -0.003                            0.009                                -0.005                                -0.006                                0.011
```

Headline combos:

```
                          combo  n_train  n_test  model_interaction_fnn_v21_beta_act3_mt  residual_train_fnn_v21_beta_act3_mt  residual_test_fnn_v21_beta_act3_mt  model_interaction_fnn_v22_beta_act3_mt_cum  residual_train_fnn_v22_beta_act3_mt_cum  residual_test_fnn_v22_beta_act3_mt_cum
           barricade + entrench     9331    2152                                   0.071                                0.411                               0.249                                       0.147                                    0.373                                   0.204
barricade + entrench + bodyslam     5315    1196                                  -0.008                                0.099                               0.736                                       0.014                                    0.102                                   0.775
        corruption + feelnopain    12789    3123                                   0.053                                0.045                              -0.080                                       0.089                                    0.014                                  -0.112
       corruption + darkembrace     8573    1995                                   0.045                               -0.061                               0.033                                       0.041                                   -0.051                                   0.036
        corruption + deadbranch     4043     964                                   0.126                                0.789                               0.921                                       0.314                                    0.635                                   0.751
```

## Combo summary

```
                     metric                    2                        8
                      model fnn_v21_beta_act3_mt fnn_v22_beta_act3_mt_cum
                  synergy_n                  111                      111
  synergy_model_interaction                0.008                    0.011
     synergy_residual_train               -0.021                   -0.033
 synergy_abs_residual_train                0.127                    0.127
  synergy_share_z_gt2_train                0.036                    0.036
synergy_share_z_lt_m2_train                0.081                    0.081
      synergy_residual_test                0.034                    0.029
  synergy_abs_residual_test                0.213                    0.200
   synergy_share_z_gt2_test                0.045                    0.036
 synergy_share_z_lt_m2_test                0.027                    0.018
                     null_n                  100                      100
     null_model_interaction               -0.000                   -0.005
        null_residual_train               -0.003                   -0.006
    null_abs_residual_train                0.101                    0.095
     null_share_z_gt2_train                0.030                    0.030
   null_share_z_lt_m2_train                0.050                    0.040
         null_residual_test                0.009                    0.011
     null_abs_residual_test                0.221                    0.226
      null_share_z_gt2_test                0.030                    0.040
    null_share_z_lt_m2_test                0.030                    0.040
```

## Card picks

```
                   model  agree_all  agree_heart_killers  agree_others  picks_a_card  decisions  cards_calibrated  mean_z2  share_abs_z_gt2  rms_rel_err
    fnn_v21_beta_act3_mt      0.409                0.551         0.405         0.680 509451.000           173.000    1.200            0.075        0.172
fnn_v22_beta_act3_mt_cum      0.413                0.561         0.410         0.668 509451.000           173.000    1.194            0.075        0.172
```
