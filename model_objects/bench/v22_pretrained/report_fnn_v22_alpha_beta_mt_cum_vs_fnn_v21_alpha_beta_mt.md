# fnn_v22_alpha_beta_mt_cum vs fnn_v21_alpha_beta_mt  (bench v22_pretrained)

## Heart head, held-out finetuning set

```
            model_version  best_ft_log_loss  best_ft_auc  best_ft_ece  best_ft_brier  best_ft_avg_precision  best_ft_best_f1
    fnn_v21_alpha_beta_mt           0.05326      0.92588      0.00052        0.01268                0.47971          0.49972
fnn_v22_alpha_beta_mt_cum           0.05301      0.92641      0.00054        0.01261                0.48425          0.50211
```

## Standard metrics on the held-out set (log loss, AUC, calibration error, Brier, precision, recall, F1 at 0.5)

```
                    model  log_loss     auc     ece   brier  avg_precision  precision@0.5  recall@0.5  f1@0.5  n_pred_pos@0.5  best_f1  precision@bestF1  recall@bestF1  threshold  recall top 5%  recall top 10%
    fnn_v21_alpha_beta_mt   0.05326 0.92588 0.00052 0.01268        0.47971        0.69448     0.30053 0.41952           10258  0.49972           0.51118        0.48876    0.24073        0.69386         0.79211
fnn_v22_alpha_beta_mt_cum   0.05301 0.92641 0.00054 0.01261        0.48425        0.70134     0.30601 0.42610           10343  0.50211           0.50521        0.49905    0.23318        0.69458         0.79338
```

## Does upgrading look like an upgrade?

Model-implied logit change from swapping a plain copy for an upgraded one on held-out decks. Most upgrades help, so fewer negative effects (especially for rare states) means upgrade states are being learned better.

```
                                       cards   n  mean effect fnn_v21_alpha_beta_mt  mean effect fnn_v22_alpha_beta_mt_cum  share <0 fnn_v21_alpha_beta_mt  share <0 fnn_v22_alpha_beta_mt_cum  corr(effects)
                                   all cards 344                              0.210                                  0.180                           0.119                               0.151          0.821
rare state (<5000 rows in plain or upgraded)  90                              0.112                                  0.073                           0.244                               0.278          0.509
                              well supported 254                              0.245                                  0.218                           0.075                               0.106          0.857
```

Lowest upgrade effects under fnn_v22_alpha_beta_mt_cum:

```
         card       char  n_train_plain  n_train_upgraded  upgrade_effect_fnn_v21_alpha_beta_mt  upgrade_effect_fnn_v22_alpha_beta_mt_cum
lessonlearned    watcher          20825              8806                                -0.704                                    -0.392
         gash     defect         120857             14687                                 0.156                                    -0.166
  forethought   ironclad           1886               613                                 0.116                                    -0.136
       unload the_silent          19827              5109                                 0.077                                    -0.135
metamorphosis   ironclad           6903              4961                                 0.010                                    -0.113
       strike   ironclad        4854417            466241                                -0.233                                    -0.109
     echoform     defect          61328             11392                                -0.174                                    -0.109
       reaper   ironclad          88886             10965                                -0.179                                    -0.092
```

Biggest changes fnn_v21_alpha_beta_mt to fnn_v22_alpha_beta_mt_cum:

```
            card       char  n_train_plain  n_train_upgraded  upgrade_effect_fnn_v21_alpha_beta_mt  upgrade_effect_fnn_v22_alpha_beta_mt_cum
   pathtovictory    watcher          17253              7451                                 0.345                                    -0.008
     outmaneuver the_silent          18850              4934                                 0.491                                     0.153
     nightterror the_silent          19941              8652                                -0.017                                     0.319
           setup the_silent           7600              3728                                 0.403                                     0.070
masterofstrategy    watcher          13146              1915                                -0.128                                     0.196
            gash     defect         120857             14687                                 0.156                                    -0.166
        accuracy the_silent          37480             15816                                 0.459                                     0.138
   lessonlearned    watcher          20825              8806                                -0.704                                    -0.392
```

## Held-out calibration of each card's plain / upgraded copy, by training support

Lower mean z2 and rms error is better calibrated. The gain from sharing should show in the low-support rows.

```
train rows of state  cells  mean z2 fnn_v21_alpha_beta_mt  rms rel err fnn_v21_alpha_beta_mt  mean z2 fnn_v22_alpha_beta_mt_cum  rms rel err fnn_v22_alpha_beta_mt_cum
             0-1000      7                          1.386                              0.284                              1.226                                  0.269
          1000-5000     80                          1.528                              0.181                              1.555                                  0.188
         5000-20000    244                          1.551                              0.113                              1.452                                  0.113
          20000-inf    333                          1.631                              0.070                              1.687                                  0.071
```

## Upgrade x combo: is a pair's interaction the same in every upgrade state?

Spread is max minus min of the pair interaction over the four plain/upgraded combinations. rarest/best near 1 means what was learned in the well-supported states carries over to the rare ones.

```
                           pair set  pairs  mean spread fnn_v21_alpha_beta_mt  mean spread fnn_v22_alpha_beta_mt_cum  strong pairs  rarest/best fnn_v21_alpha_beta_mt  rarest/best fnn_v22_alpha_beta_mt_cum
                          all pairs    190                              0.041                                  0.043            91                              0.681                                  1.140
rarest state combination <5000 rows    187                              0.041                                  0.042            89                              0.675                                  1.163
                  rarest <1000 rows    119                              0.036                                  0.033            54                              0.589                                  1.172
```

Pairs whose spread changed most:

```
                    combo  n_best  n_rarest  spread fnn_v21_alpha_beta_mt  spread fnn_v22_alpha_beta_mt_cum  best fnn_v22_alpha_beta_mt_cum  rarest fnn_v22_alpha_beta_mt_cum
    accuracy + bladedance    3285      2509                         0.034                             0.215                           0.196                             0.279
     barricade + entrench    3439      1605                         0.065                             0.235                           0.425                             0.302
     barricade + bodyslam    6038      3120                         0.051                             0.174                           0.349                             0.263
accuracy + cloakanddagger    3420      2448                         0.087                             0.203                           0.304                             0.165
accuracy + infiniteblades    3068      1016                         0.053                             0.162                           0.154                             0.227
       creativeai + storm    1701       928                         0.057                             0.149                           0.206                             0.355
  metallicize + barricade    3203      1385                         0.029                             0.116                           0.079                             0.126
 catalyst + bouncingflask    8000      3615                         0.118                             0.034                           0.121                             0.140
```

## Combos: model interaction and what it leaves unexplained (logit)

Residual > 0 means the model undervalues the combo. Compare against the null controls for the noise level.

```
                               set   n  interaction fnn_v21_alpha_beta_mt  resid train fnn_v21_alpha_beta_mt  resid test fnn_v21_alpha_beta_mt  interaction fnn_v22_alpha_beta_mt_cum  resid train fnn_v22_alpha_beta_mt_cum  resid test fnn_v22_alpha_beta_mt_cum
                     listed combos 111                              0.020                             -0.027                             0.029                                  0.028                                 -0.033                                 0.030
listed pairs with upgradable cards  90                              0.009                             -0.039                            -0.007                                  0.016                                 -0.046                                -0.005
         listed, relics or triples  21                              0.069                              0.025                             0.187                                  0.084                                  0.022                                 0.178
                     null controls 100                              0.003                             -0.011                            -0.000                                 -0.001                                 -0.013                                 0.000
```

Headline combos:

```
                          combo  n_train  n_test  model_interaction_fnn_v21_alpha_beta_mt  residual_train_fnn_v21_alpha_beta_mt  residual_test_fnn_v21_alpha_beta_mt  model_interaction_fnn_v22_alpha_beta_mt_cum  residual_train_fnn_v22_alpha_beta_mt_cum  residual_test_fnn_v22_alpha_beta_mt_cum
           barricade + entrench     9331    2152                                    0.087                                 0.439                                0.254                                        0.188                                     0.345                                    0.149
barricade + entrench + bodyslam     5315    1196                                    0.007                                 0.084                                0.783                                       -0.000                                     0.132                                    0.742
        corruption + feelnopain    12789    3123                                    0.089                                 0.019                               -0.151                                        0.115                                     0.033                                   -0.123
       corruption + darkembrace     8573    1995                                    0.047                                -0.043                                0.074                                        0.071                                    -0.023                                    0.098
        corruption + deadbranch     4043     964                                    0.600                                 0.329                                0.489                                        0.752                                     0.254                                    0.369
```

## Combo summary

```
                     metric                     4                         9
                      model fnn_v21_alpha_beta_mt fnn_v22_alpha_beta_mt_cum
                  synergy_n                   111                       111
  synergy_model_interaction                 0.020                     0.028
     synergy_residual_train                -0.027                    -0.033
 synergy_abs_residual_train                 0.127                     0.119
  synergy_share_z_gt2_train                 0.036                     0.036
synergy_share_z_lt_m2_train                 0.072                     0.072
      synergy_residual_test                 0.029                     0.030
  synergy_abs_residual_test                 0.212                     0.211
   synergy_share_z_gt2_test                 0.036                     0.027
 synergy_share_z_lt_m2_test                 0.018                     0.027
                     null_n                   100                       100
     null_model_interaction                 0.003                    -0.001
        null_residual_train                -0.011                    -0.013
    null_abs_residual_train                 0.096                     0.092
     null_share_z_gt2_train                 0.010                     0.010
   null_share_z_lt_m2_train                 0.050                     0.020
         null_residual_test                -0.000                     0.000
     null_abs_residual_test                 0.228                     0.225
      null_share_z_gt2_test                 0.030                     0.020
    null_share_z_lt_m2_test                 0.030                     0.030
```

## Card picks

```
                    model  agree_all  agree_heart_killers  agree_others  picks_a_card  decisions  cards_calibrated  mean_z2  share_abs_z_gt2  rms_rel_err
    fnn_v21_alpha_beta_mt      0.418                0.568         0.415         0.696 509451.000           173.000    1.148            0.081        0.168
fnn_v22_alpha_beta_mt_cum      0.416                0.572         0.412         0.680 509451.000           173.000    1.107            0.069        0.166
```
