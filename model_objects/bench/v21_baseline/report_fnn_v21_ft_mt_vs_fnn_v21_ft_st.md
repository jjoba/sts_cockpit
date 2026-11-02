# fnn_v21_ft_mt vs fnn_v21_ft_st  (bench v21_baseline)

## Heart head, held-out finetuning set

```
model_version  best_ft_log_loss  best_ft_auc  best_ft_avg_precision  best_ft_ece
fnn_v21_ft_st             0.055        0.919                  0.461        0.001
fnn_v21_ft_mt             0.054        0.922                  0.464        0.001
```

## Does upgrading look like an upgrade?

Model-implied logit change from swapping a plain copy for an upgraded one on held-out decks. Most upgrades help, so fewer negative effects (especially for rare states) means upgrade states are being learned better.

```
                                       cards   n  mean effect fnn_v21_ft_st  mean effect fnn_v21_ft_mt  share <0 fnn_v21_ft_st  share <0 fnn_v21_ft_mt  corr(effects)
                                   all cards 344                      0.157                      0.201                   0.244                   0.177          0.885
rare state (<5000 rows in plain or upgraded)  90                      0.065                      0.096                   0.389                   0.333          0.881
                              well supported 254                      0.190                      0.238                   0.193                   0.122          0.877
```

Lowest upgrade effects under fnn_v21_ft_mt:

```
           card     char  n_train_plain  n_train_upgraded  upgrade_effect_fnn_v21_ft_st  upgrade_effect_fnn_v21_ft_mt
  lessonlearned  watcher          20825              8806                        -0.583                        -0.889
   darkshackles ironclad           8240              1220                        -0.487                        -0.391
  metamorphosis ironclad           6903              4961                        -0.164                        -0.241
secrettechnique  watcher           4650              1061                        -0.331                        -0.227
       echoform   defect          61328             11392                        -0.382                        -0.215
    panicbutton ironclad           2604               348                        -0.089                        -0.215
         reaper ironclad          88886             10965                        -0.326                        -0.200
        glacier   defect         124791             18854                        -0.322                        -0.198
```

Biggest changes fnn_v21_ft_st to fnn_v21_ft_mt:

```
         card     char  n_train_plain  n_train_upgraded  upgrade_effect_fnn_v21_ft_st  upgrade_effect_fnn_v21_ft_mt
   battlehymn  watcher          28747             14386                        -0.442                        -0.003
 carvereality  watcher          23824              4955                        -0.230                         0.136
   twinstrike ironclad         161918             28180                        -0.150                         0.178
       lockon   defect          38580             11987                        -0.042                         0.286
        chill   defect          76308             12145                        -0.118                         0.198
lessonlearned  watcher          20825              8806                        -0.583                        -0.889
     ironwave ironclad         102374             21627                        -0.067                         0.236
   multi-cast   defect          13882              4870                        -0.103                         0.191
```

## Held-out calibration of each card's plain / upgraded copy, by training support

Lower mean z2 and rms error is better calibrated. The gain from sharing should show in the low-support rows.

```
train rows of state  cells  mean z2 fnn_v21_ft_st  rms rel err fnn_v21_ft_st  mean z2 fnn_v21_ft_mt  rms rel err fnn_v21_ft_mt
             0-1000      7                  1.423                      0.276                  1.612                      0.269
          1000-5000     80                  1.441                      0.169                  1.542                      0.167
         5000-20000    244                  1.255                      0.095                  1.822                      0.110
          20000-inf    333                  1.117                      0.055                  2.045                      0.070
```

## Upgrade x combo: is a pair's interaction the same in every upgrade state?

Spread is max minus min of the pair interaction over the four plain/upgraded combinations. rarest/best near 1 means what was learned in the well-supported states carries over to the rare ones.

```
                           pair set  pairs  mean spread fnn_v21_ft_st  mean spread fnn_v21_ft_mt  strong pairs  rarest/best fnn_v21_ft_st  rarest/best fnn_v21_ft_mt
                          all pairs    190                      0.018                      0.032             9                      0.347                      0.817
rarest state combination <5000 rows    187                      0.018                      0.031             9                      0.347                      0.817
                  rarest <1000 rows    119                      0.015                      0.026             3                      0.485                      0.871
```

Pairs whose spread changed most:

```
                          combo  n_best  n_rarest  spread fnn_v21_ft_st  spread fnn_v21_ft_mt  best fnn_v21_ft_mt  rarest fnn_v21_ft_mt
           battlehymn + protect    2032       726                 0.006                 0.078               0.081                 0.003
          evolve + powerthrough    6580      2104                 0.009                 0.077               0.072                 0.020
   bouncingflask + noxiousfumes    8255      5216                 0.009                 0.070              -0.307                -0.258
 corpseexplosion + noxiousfumes    9833      1720                 0.016                 0.076              -0.069                -0.116
   bouncingflask + piercingwail   10210      1428                 0.021                 0.080               0.127                 0.047
       catalyst + bouncingflask    8000      3615                 0.044                 0.102              -0.056                 0.029
electrodynamics + balllightning   17596      1427                 0.014                 0.070              -0.032                -0.018
        catalyst + noxiousfumes   11209      5405                 0.010                 0.066              -0.023                 0.033
```

## Combos: model interaction and what it leaves unexplained (logit)

Residual > 0 means the model undervalues the combo. Compare against the null controls for the noise level.

```
                               set   n  interaction fnn_v21_ft_st  resid train fnn_v21_ft_st  resid test fnn_v21_ft_st  interaction fnn_v21_ft_mt  resid train fnn_v21_ft_mt  resid test fnn_v21_ft_mt
                     listed combos 111                      0.001                      0.006                     0.063                      0.004                     -0.016                     0.041
listed pairs with upgradable cards  90                     -0.000                     -0.017                     0.019                     -0.003                     -0.037                    -0.006
         listed, relics or triples  21                      0.005                      0.105                     0.252                      0.034                      0.076                     0.239
                     null controls 100                      0.002                      0.002                     0.016                      0.006                      0.002                     0.014
```

Headline combos:

```
                          combo  n_train  n_test  model_interaction_fnn_v21_ft_st  residual_train_fnn_v21_ft_st  residual_test_fnn_v21_ft_st  model_interaction_fnn_v21_ft_mt  residual_train_fnn_v21_ft_mt  residual_test_fnn_v21_ft_mt
           barricade + entrench     9331    2152                            0.025                         0.457                        0.341                            0.050                         0.412                        0.269
barricade + entrench + bodyslam     5315    1196                            0.001                         0.055                        0.724                            0.002                         0.116                        0.755
        corruption + feelnopain    12789    3123                            0.028                         0.095                       -0.041                            0.004                         0.052                       -0.079
       corruption + darkembrace     8573    1995                            0.008                         0.019                        0.112                           -0.030                        -0.059                        0.038
        corruption + deadbranch     4043     964                            0.022                         0.895                        1.083                            0.076                         0.770                        0.934
```

## Combo summary

```
                     metric             0             1
                      model fnn_v21_ft_st fnn_v21_ft_mt
                  synergy_n           111           111
  synergy_model_interaction         0.001         0.004
     synergy_residual_train         0.006        -0.016
 synergy_abs_residual_train         0.132         0.131
  synergy_share_z_gt2_train         0.072         0.054
synergy_share_z_lt_m2_train         0.090         0.108
      synergy_residual_test         0.063         0.041
  synergy_abs_residual_test         0.235         0.223
   synergy_share_z_gt2_test         0.072         0.072
 synergy_share_z_lt_m2_test         0.018         0.036
                     null_n           100           100
     null_model_interaction         0.002         0.006
        null_residual_train         0.002         0.002
    null_abs_residual_train         0.104         0.103
     null_share_z_gt2_train         0.030         0.020
   null_share_z_lt_m2_train         0.040         0.060
         null_residual_test         0.016         0.014
     null_abs_residual_test         0.228         0.220
      null_share_z_gt2_test         0.040         0.030
    null_share_z_lt_m2_test         0.030         0.030
```

## Card picks

```
        model  agree_all  agree_heart_killers  agree_others  picks_a_card  decisions  cards_calibrated  mean_z2  share_abs_z_gt2  rms_rel_err
fnn_v21_ft_st      0.402                0.552         0.398         0.707 509451.000           173.000    1.129            0.064        0.176
fnn_v21_ft_mt      0.417                0.554         0.414         0.705 509451.000           173.000    1.282            0.087        0.178
```
