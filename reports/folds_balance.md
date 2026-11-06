# Fold balance report

Dataset: 3480 câu, 444 compound  
Folds: 5  
Repeats: seed 42, 43, 44

## Số compound theo tầng

|     Stratum     |             |  Số compound |
| 0_0 | 0_1 | 0_2 |             | 63 | 43 | 42 |
| 1_0 | 1_1 | 1_2 |     ---->   | 44 | 60 | 44 |
| 2_0 | 2_1 | 2_2 |             | 41 | 47 | 60 |

## Repeat 0 — seed 42

| Fold | Compound valid | Câu valid | Compound train | Câu train | Compound valid có modifier/head xuất hiện trong train | Số câu/compound trong valid |
|  0   |       89       |    707    |       355      |    2773   | ... | ... | .../... (...) | min–max; hoặc median |
|  1   |       89       |    690    |       355      |    2790   | ... | ... | ... | ... |
|  2   |       89       |    722    |       355      |    2758   | ... | ... | ... | ... |
|  3   |       88       |    672    |       356      |    2808   | ... | ... | ... | ... |
|  4   |       89       |    689    |       356      |    2791   | ... | ... | ... | ... |

## Repeat 1 — seed 43

| Fold | Compound valid | Câu valid | Compound train | Câu train | Compound valid có modifier/head xuất hiện trong train | Số câu/compound trong valid |
|  0   |       89       |    671    |       355      |    2809   | ... | ... | .../... (...) | min–max; hoặc median |
|  1   |       88       |    715    |       356      |    2765   | ... | ... | ... | ... |
|  2   |       88       |    644    |       356      |    2836   | ... | ... | ... | ... |
|  3   |       88       |    686    |       356      |    2794   | ... | ... | ... | ... |
|  4   |       91       |    764    |       353      |    2716   | ... | ... | ... | ... |

## Repeat 2 — seed 44

| Fold | Compound valid | Câu valid | Compound train | Câu train | Compound valid có modifier/head xuất hiện trong train | Số câu/compound trong valid |
|  0   |       89       |    710    |       355      |    2770   | ... | ... | .../... (...) | min–max; hoặc median |
|  1   |       89       |    697    |       355      |    2783   | ... | ... | ... | ... |
|  2   |       88       |    696    |       356      |    2784   | ... | ... | ... | ... |
|  3   |       89       |    682    |       355      |    2798   | ... | ... | ... | ... |
|  4   |       89       |    695    |       355      |    2785   | ... | ... | ... | ... |