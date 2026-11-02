# fnn_v22_ft_mt_cum vs fnn_v21_ft_mt  (bench v22_mt_cum)

## Heart head, held-out finetuning set

```
    model_version  best_ft_log_loss  best_ft_auc  best_ft_ece  best_ft_brier  best_ft_avg_precision  best_ft_best_f1
    fnn_v21_ft_mt           0.05430      0.92186      0.00092        0.01289                0.46432          0.49075
fnn_v22_ft_mt_cum           0.05401      0.92270      0.00066        0.01282                0.46854          0.49423
```

## Standard metrics on the held-out set (log loss, AUC, calibration error, Brier, precision, recall, F1 at 0.5)

```
            model  log_loss     auc     ece   brier  avg_precision  precision@0.5  recall@0.5  f1@0.5  n_pred_pos@0.5  best_f1  precision@bestF1  recall@bestF1  threshold  recall top 5%  recall top 10%
    fnn_v21_ft_mt   0.05430 0.92186 0.00092 0.01289        0.46432        0.67964     0.30205 0.41822           10535  0.49075           0.50000        0.48184    0.25307        0.68454         0.78456
fnn_v22_ft_mt_cum   0.05401 0.92270 0.00066 0.01282        0.46854        0.67981     0.30479 0.42088           10628  0.49423           0.50312        0.48564    0.25000        0.68703         0.78696
```

## Does upgrading look like an upgrade?

Model-implied logit change from swapping a plain copy for an upgraded one on held-out decks. Most upgrades help, so fewer negative effects (especially for rare states) means upgrade states are being learned better.

```
                                       cards   n  mean effect fnn_v21_ft_mt  mean effect fnn_v22_ft_mt_cum  share <0 fnn_v21_ft_mt  share <0 fnn_v22_ft_mt_cum  corr(effects)
                                   all cards 344                      0.201                          0.180                   0.177                       0.180          0.832
rare state (<5000 rows in plain or upgraded)  90                      0.096                          0.054                   0.333                       0.356          0.506
                              well supported 254                      0.238                          0.225                   0.122                       0.118          0.886
```

Lowest upgrade effects under fnn_v22_ft_mt_cum:

```
         card       char  n_train_plain  n_train_upgraded  upgrade_effect_fnn_v21_ft_mt  upgrade_effect_fnn_v22_ft_mt_cum
lessonlearned    watcher          20825              8806                        -0.889                            -0.539
metamorphosis   ironclad           6903              4961                        -0.241                            -0.295
       unload the_silent          19827              5109                         0.023                            -0.224
  forethought   ironclad           1886               613                        -0.053                            -0.167
         trip    watcher          10035              4809                        -0.152                            -0.166
       reaper   ironclad          88886             10965                        -0.200                            -0.130
      rampage   ironclad          79206             42094                        -0.095                            -0.130
    reprogram     defect           5789              1308                         0.331                            -0.113
```

Biggest changes fnn_v21_ft_mt to fnn_v22_ft_mt_cum:

```
         card       char  n_train_plain  n_train_upgraded  upgrade_effect_fnn_v21_ft_mt  upgrade_effect_fnn_v22_ft_mt_cum
        setup the_silent           7600              3728                         0.452                            -0.062
    reprogram     defect           5789              1308                         0.331                            -0.113
  outmaneuver the_silent          18850              4934                         0.628                             0.186
wreathofflame    watcher           5456              2086                         0.367                            -0.050
        alpha    watcher           7762              4163                         0.386                            -0.009
 darkshackles   ironclad           8240              1220                        -0.391                            -0.005
        study    watcher           8310              5170                         0.527                             0.168
    aggregate     defect           5367              1410                         0.385                             0.027
```

## Held-out calibration of each card's plain / upgraded copy, by training support

Lower mean z2 and rms error is better calibrated. The gain from sharing should show in the low-support rows.

```
train rows of state  cells  mean z2 fnn_v21_ft_mt  rms rel err fnn_v21_ft_mt  mean z2 fnn_v22_ft_mt_cum  rms rel err fnn_v22_ft_mt_cum
             0-1000      7                  1.612                      0.269                      1.517                          0.257
          1000-5000     80                  1.542                      0.167                      1.473                          0.175
         5000-20000    244                  1.822                      0.110                      1.610                          0.108
          20000-inf    333                  2.045                      0.070                      1.668                          0.064
```

## Upgrade x combo: is a pair's interaction the same in every upgrade state?

Spread is max minus min of the pair interaction over the four plain/upgraded combinations. rarest/best near 1 means what was learned in the well-supported states carries over to the rare ones.

```
                           pair set  pairs  mean spread fnn_v21_ft_mt  mean spread fnn_v22_ft_mt_cum  strong pairs  rarest/best fnn_v21_ft_mt  rarest/best fnn_v22_ft_mt_cum
                          all pairs    190                      0.032                          0.033            62                      0.552                          1.062
rarest state combination <5000 rows    187                      0.031                          0.032            60                      0.535                          1.068
                  rarest <1000 rows    119                      0.026                          0.027            25                      0.498                          1.158
```

Pairs whose spread changed most:

```
                     combo  n_best  n_rarest  spread fnn_v21_ft_mt  spread fnn_v22_ft_mt_cum  best fnn_v22_ft_mt_cum  rarest fnn_v22_ft_mt_cum
     accuracy + bladedance    3285      2509                 0.014                     0.103                   0.045                     0.090
  afterimage + wraithformv    1965       380                 0.060                     0.142                  -0.138                    -0.125
 accuracy + cloakanddagger    3420      2448                 0.027                     0.109                   0.154                     0.093
barricade + swordboomerang    2314       876                 0.011                     0.092                  -0.102                    -0.194
  catalyst + welllaidplans    4635      1855                 0.071                     0.139                  -0.075                    -0.094
      barricade + entrench    3439      1605                 0.017                     0.083                   0.174                     0.118
    envenom + noxiousfumes    2527      1309                 0.033                     0.099                  -0.183                    -0.198
  catalyst + bouncingflask    8000      3615                 0.102                     0.039                   0.060                     0.099
```

## Combos: model interaction and what it leaves unexplained (logit)

Residual > 0 means the model undervalues the combo. Compare against the null controls for the noise level.

```
                               set   n  interaction fnn_v21_ft_mt  resid train fnn_v21_ft_mt  resid test fnn_v21_ft_mt  interaction fnn_v22_ft_mt_cum  resid train fnn_v22_ft_mt_cum  resid test fnn_v22_ft_mt_cum
                     listed combos 111                      0.004                     -0.016                     0.041                          0.008                         -0.024                         0.036
listed pairs with upgradable cards  90                     -0.003                     -0.037                    -0.006                          0.000                         -0.045                        -0.008
         listed, relics or triples  21                      0.034                      0.076                     0.239                          0.041                          0.066                         0.227
                     null controls 100                      0.006                      0.002                     0.014                          0.003                          0.000                         0.014
```

Headline combos:

```
                          combo  n_train  n_test  model_interaction_fnn_v21_ft_mt  residual_train_fnn_v21_ft_mt  residual_test_fnn_v21_ft_mt  model_interaction_fnn_v22_ft_mt_cum  residual_train_fnn_v22_ft_mt_cum  residual_test_fnn_v22_ft_mt_cum
           barricade + entrench     9331    2152                            0.050                         0.412                        0.269                                0.090                             0.357                            0.187
barricade + entrench + bodyslam     5315    1196                            0.002                         0.116                        0.755                                0.000                             0.117                            0.770
        corruption + feelnopain    12789    3123                            0.004                         0.052                       -0.079                                0.036                             0.031                           -0.110
       corruption + darkembrace     8573    1995                           -0.030                        -0.059                        0.038                               -0.014                            -0.053                            0.046
        corruption + deadbranch     4043     964                            0.076                         0.770                        0.934                                0.183                             0.700                            0.854
```

## Combo summary

```
                     metric             0                 1
                      model fnn_v21_ft_mt fnn_v22_ft_mt_cum
                  synergy_n           111               111
  synergy_model_interaction         0.004             0.008
     synergy_residual_train        -0.016            -0.024
 synergy_abs_residual_train         0.131             0.127
  synergy_share_z_gt2_train         0.054             0.036
synergy_share_z_lt_m2_train         0.108             0.072
      synergy_residual_test         0.041             0.036
  synergy_abs_residual_test         0.223             0.214
   synergy_share_z_gt2_test         0.072             0.054
 synergy_share_z_lt_m2_test         0.036             0.027
                     null_n           100               100
     null_model_interaction         0.006             0.003
        null_residual_train         0.002             0.000
    null_abs_residual_train         0.103             0.100
     null_share_z_gt2_train         0.020             0.030
   null_share_z_lt_m2_train         0.060             0.050
         null_residual_test         0.014             0.014
     null_abs_residual_test         0.220             0.222
      null_share_z_gt2_test         0.030             0.030
    null_share_z_lt_m2_test         0.030             0.030
```

## Card picks

```
            model  agree_all  agree_heart_killers  agree_others  picks_a_card  decisions  cards_calibrated  mean_z2  share_abs_z_gt2  rms_rel_err
    fnn_v21_ft_mt      0.417                0.554         0.414         0.705 509451.000           173.000    1.282            0.087        0.178
fnn_v22_ft_mt_cum      0.421                0.560         0.418         0.718 509451.000           173.000    1.301            0.087        0.178
```
