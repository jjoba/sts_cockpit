# fnn_v22_beta_mt_cum vs fnn_v21_beta_mt  (bench v22_pretrained)

## Heart head, held-out finetuning set

```
      model_version  best_ft_log_loss  best_ft_auc  best_ft_ece  best_ft_brier  best_ft_avg_precision  best_ft_best_f1
    fnn_v21_beta_mt           0.05361      0.92468      0.00029        0.01276                0.47396          0.49558
fnn_v22_beta_mt_cum           0.05330      0.92559      0.00063        0.01268                0.47901          0.49866
```

## Standard metrics on the held-out set (log loss, AUC, calibration error, Brier, precision, recall, F1 at 0.5)

```
              model  log_loss     auc     ece   brier  avg_precision  precision@0.5  recall@0.5  f1@0.5  n_pred_pos@0.5  best_f1  precision@bestF1  recall@bestF1  threshold  recall top 5%  recall top 10%
    fnn_v21_beta_mt   0.05361 0.92468 0.00029 0.01276        0.47396        0.68971     0.30015 0.41827           10316  0.49558           0.51932        0.47391    0.25866        0.69099         0.79000
fnn_v22_beta_mt_cum   0.05330 0.92559 0.00063 0.01268        0.47901        0.68546     0.31183 0.42866           10784  0.49866           0.51722        0.48138    0.26521        0.69428         0.79274
```

## Does upgrading look like an upgrade?

Model-implied logit change from swapping a plain copy for an upgraded one on held-out decks. Most upgrades help, so fewer negative effects (especially for rare states) means upgrade states are being learned better.

```
                                       cards   n  mean effect fnn_v21_beta_mt  mean effect fnn_v22_beta_mt_cum  share <0 fnn_v21_beta_mt  share <0 fnn_v22_beta_mt_cum  corr(effects)
                                   all cards 344                        0.208                            0.185                     0.137                         0.154          0.827
rare state (<5000 rows in plain or upgraded)  90                        0.108                            0.074                     0.256                         0.278          0.550
                              well supported 254                        0.243                            0.224                     0.094                         0.110          0.863
```

Lowest upgrade effects under fnn_v22_beta_mt_cum:

```
         card       char  n_train_plain  n_train_upgraded  upgrade_effect_fnn_v21_beta_mt  upgrade_effect_fnn_v22_beta_mt_cum
lessonlearned    watcher          20825              8806                          -0.828                              -0.459
       unload the_silent          19827              5109                           0.068                              -0.192
         gash     defect         120857             14687                           0.246                              -0.184
  forethought   ironclad           1886               613                           0.036                              -0.184
metamorphosis   ironclad           6903              4961                          -0.074                              -0.140
      rampage   ironclad          79206             42094                          -0.104                              -0.120
       reaper   ironclad          88886             10965                          -0.167                              -0.120
    reprogram     defect           5789              1308                           0.196                              -0.114
```

Biggest changes fnn_v21_beta_mt to fnn_v22_beta_mt_cum:

```
         card       char  n_train_plain  n_train_upgraded  upgrade_effect_fnn_v21_beta_mt  upgrade_effect_fnn_v22_beta_mt_cum
         gash     defect         120857             14687                           0.246                              -0.184
pathtovictory    watcher          17253              7451                           0.373                              -0.008
lessonlearned    watcher          20825              8806                          -0.828                              -0.459
     accuracy the_silent          37480             15816                           0.525                               0.160
        setup the_silent           7600              3728                           0.410                               0.051
  outmaneuver the_silent          18850              4934                           0.551                               0.222
 darkshackles   ironclad           8240              1220                          -0.332                              -0.018
    reprogram     defect           5789              1308                           0.196                              -0.114
```

## Held-out calibration of each card's plain / upgraded copy, by training support

Lower mean z2 and rms error is better calibrated. The gain from sharing should show in the low-support rows.

```
train rows of state  cells  mean z2 fnn_v21_beta_mt  rms rel err fnn_v21_beta_mt  mean z2 fnn_v22_beta_mt_cum  rms rel err fnn_v22_beta_mt_cum
             0-1000      7                    1.484                        0.290                        1.262                            0.261
          1000-5000     80                    1.528                        0.178                        1.500                            0.176
         5000-20000    244                    1.502                        0.108                        1.421                            0.103
          20000-inf    333                    1.288                        0.059                        1.331                            0.057
```

## Upgrade x combo: is a pair's interaction the same in every upgrade state?

Spread is max minus min of the pair interaction over the four plain/upgraded combinations. rarest/best near 1 means what was learned in the well-supported states carries over to the rare ones.

```
                           pair set  pairs  mean spread fnn_v21_beta_mt  mean spread fnn_v22_beta_mt_cum  strong pairs  rarest/best fnn_v21_beta_mt  rarest/best fnn_v22_beta_mt_cum
                          all pairs    190                        0.040                            0.043            81                        0.574                            1.088
rarest state combination <5000 rows    187                        0.039                            0.042            79                        0.565                            1.095
                  rarest <1000 rows    119                        0.035                            0.034            49                        0.539                            1.108
```

Pairs whose spread changed most:

```
                         combo  n_best  n_rarest  spread fnn_v21_beta_mt  spread fnn_v22_beta_mt_cum  best fnn_v22_beta_mt_cum  rarest fnn_v22_beta_mt_cum
          barricade + entrench    3439      1605                   0.023                       0.208                     0.344                       0.238
         accuracy + bladedance    3285      2509                   0.018                       0.158                     0.153                       0.219
          barricade + bodyslam    6038      3120                   0.048                       0.186                     0.310                       0.197
      afterimage + wraithformv    1965       380                   0.049                       0.156                    -0.169                      -0.161
       metallicize + barricade    3203      1385                   0.026                       0.126                     0.079                       0.136
     accuracy + cloakanddagger    3420      2448                   0.047                       0.143                     0.219                       0.116
      catalyst + welllaidplans    4635      1855                   0.089                       0.168                    -0.142                      -0.151
cripplingpoison + noxiousfumes    8485      4618                   0.034                       0.113                    -0.150                      -0.094
```

## Combos: model interaction and what it leaves unexplained (logit)

Residual > 0 means the model undervalues the combo. Compare against the null controls for the noise level.

```
                               set   n  interaction fnn_v21_beta_mt  resid train fnn_v21_beta_mt  resid test fnn_v21_beta_mt  interaction fnn_v22_beta_mt_cum  resid train fnn_v22_beta_mt_cum  resid test fnn_v22_beta_mt_cum
                     listed combos 111                        0.007                       -0.021                       0.038                            0.013                           -0.025                           0.039
listed pairs with upgradable cards  90                       -0.002                       -0.038                      -0.003                            0.005                           -0.041                          -0.000
         listed, relics or triples  21                        0.046                        0.055                       0.211                            0.053                            0.044                           0.206
                     null controls 100                        0.000                       -0.003                       0.006                           -0.002                           -0.007                           0.009
```

Headline combos:

```
                          combo  n_train  n_test  model_interaction_fnn_v21_beta_mt  residual_train_fnn_v21_beta_mt  residual_test_fnn_v21_beta_mt  model_interaction_fnn_v22_beta_mt_cum  residual_train_fnn_v22_beta_mt_cum  residual_test_fnn_v22_beta_mt_cum
           barricade + entrench     9331    2152                              0.055                           0.433                          0.259                                  0.129                               0.358                              0.191
barricade + entrench + bodyslam     5315    1196                              0.005                           0.082                          0.777                                  0.013                               0.127                              0.791
        corruption + feelnopain    12789    3123                              0.064                           0.036                         -0.102                                  0.072                               0.036                             -0.116
       corruption + darkembrace     8573    1995                              0.014                          -0.052                          0.037                                  0.016                              -0.037                              0.073
        corruption + deadbranch     4043     964                              0.202                           0.738                          0.892                                  0.406                               0.563                              0.706
```

## Combo summary

```
                     metric               3                   7
                      model fnn_v21_beta_mt fnn_v22_beta_mt_cum
                  synergy_n             111                 111
  synergy_model_interaction           0.007               0.013
     synergy_residual_train          -0.021              -0.025
 synergy_abs_residual_train           0.126               0.120
  synergy_share_z_gt2_train           0.036               0.036
synergy_share_z_lt_m2_train           0.072               0.072
      synergy_residual_test           0.038               0.039
  synergy_abs_residual_test           0.216               0.208
   synergy_share_z_gt2_test           0.036               0.045
 synergy_share_z_lt_m2_test           0.018               0.027
                     null_n             100                 100
     null_model_interaction           0.000              -0.002
        null_residual_train          -0.003              -0.007
    null_abs_residual_train           0.101               0.094
     null_share_z_gt2_train           0.020               0.010
   null_share_z_lt_m2_train           0.040               0.040
         null_residual_test           0.006               0.009
     null_abs_residual_test           0.228               0.226
      null_share_z_gt2_test           0.040               0.040
    null_share_z_lt_m2_test           0.030               0.030
```

## Card picks

```
              model  agree_all  agree_heart_killers  agree_others  picks_a_card  decisions  cards_calibrated  mean_z2  share_abs_z_gt2  rms_rel_err
    fnn_v21_beta_mt      0.416                0.559         0.412         0.700 509451.000           173.000    1.176            0.081        0.169
fnn_v22_beta_mt_cum      0.419                0.572         0.415         0.709 509451.000           173.000    1.149            0.069        0.171
```
