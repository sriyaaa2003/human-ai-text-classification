# Classifying Human-AI Collaboration Patterns in Text

Six-class classification of how a text was written, from fully human to fully AI, using stylometric features, a hierarchical classifier and association rules.

## Classes
| ID | Class |
|----|-------|
| 0 | Fully Human |
| 1 | Human-AI Polished |
| 2 | AI-AI Humanized |
| 3 | Human-AI Continued |
| 4 | Deeply Mixed |
| 5 | AI-Human Edited |

## What I did
1. **`dataexploration.ipynb`:** looked at 288,918 train and 72,661 test articles. The given split was uneven (class 3 was 3.7% of train but 51.2% of test).
2. **`dataprep.ipynb`:** merged all 361,579 articles, made a new stratified split (289,263 train / 72,316 test) and extracted 15 stylometric features with spaCy (burstiness, part-of-speech ratios, syntactic complexity, pronouns, contractions, hapax legomena, formal transitions, passive voice, n-gram repetition, style breaks, lexical density and more).
3. **`hier.ipynb`:** built a hierarchical XGBoost classifier with SMOTE for the rare class 5, compared it with Gradient Boosting and Random Forest, and made a weighted-vote ensemble.
4. **`binandpm.ipynb`:** binned the features with decision-tree splits and mined association rules per class with my own Apriori implementation, giving 149 class-specific rules.
5. **`rules+hier.ipynb`:** blended the rule scores with the hierarchical model.

## Results (72,316 held-out articles)
| Model | Accuracy | Macro F1 |
|-------|---------:|---------:|
| Hierarchical classifier | 62.15% | 0.4895 |
| Random Forest (tuned) | 59.28% | 0.5162 |
| Ensemble (hierarchical + Gradient Boosting + Random Forest) | 62.19% | 0.5445 |
| Hierarchical + mined rules | 62.14% | 0.4897 |

- The mined rules explain the classes. For example fully human text has high burstiness and few formal transitions, while AI-AI humanized text has low burstiness and high lexical density.
- The two rarest classes score lowest (class 5 F1 about 0.07 for the hierarchical model).

## Tech stack
Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn (SMOTE), spaCy, matplotlib, seaborn, Jupyter.
