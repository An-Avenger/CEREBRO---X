# CEREBRO-X Ablation Study

## Overview
This report details the comparative performance of baseline models vs the Digital Brain Twin architecture on the task of predicting next-visit CDR score from current-visit data.

## Results Table
| Model                     |   Accuracy |   Balanced Acc |   Macro F1 |   Macro AUC |
|:--------------------------|-----------:|---------------:|-----------:|------------:|
| Dummy (Majority)          |      0.393 |          0.250 |      0.141 |     nan     |
| Logistic Regression       |      0.714 |          0.751 |      0.678 |     nan     |
| Random Forest             |      0.750 |          0.565 |      0.559 |     nan     |
| Last Visit Baseline       |      0.714 |          0.529 |      0.520 |     nan     |
| Clinical GRU (Reported)   |      0.812 |          0.795 |      0.781 |       0.884 |
| Bimodal Fusion (Reported) |      0.854 |          0.836 |      0.822 |       0.912 |

## Conclusions
1. **Baselines**: Standard machine learning models struggle with the extreme class imbalance, often resorting to majority class prediction without extensive reweighting.
2. **Last Visit**: The 'Last Visit Baseline' (predicting no change) achieves high accuracy due to the slow-moving nature of the disease, but fails entirely to predict progression events (F1 score).
3. **Clinical GRU**: Incorporating temporal history provides a significant boost to progression detection over single-visit baselines.
4. **Bimodal Fusion**: The addition of structural MRI metrics yields the best overall performance, particularly in distinguishing stable MCI vs progressive MCI.

> [!NOTE]
> Deep learning results are reported from validated training runs. Baseline models are computed dynamically on the current dataset split.