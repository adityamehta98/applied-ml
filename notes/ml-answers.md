# Core ML interview answers (week 1)

Each answer: claim, why, example from a build, trade-off. Say them aloud.

## 1. Train, validation, test: what is each for?
Train is what the model learns from, validation is what I tune and choose models against,
and test is the sealed final exam I touch once. The validation set gets "used up" as I make
decisions against it, so a separate untouched test set gives the one honest estimate. On
applied-ml my load_split seals 20 percent as test and I only score on it at the very end.
Trade-off: with little data, a single validation split is noisy, so I use cross-validation
instead and keep only the test set aside.

## 2. What is data leakage and how do you prevent it?
Leakage is any information from validation or test rows reaching the model during training,
so it looks great offline and fails live. The usual source is fitting preprocessing on all
rows before splitting. I proved this on Day 3: selecting features on the whole dataset made a
model score about 0.79 on pure noise, versus about 0.55 when selection lived inside the
Pipeline. Prevention: put every learned step in a Pipeline so cross-validation refits it on
each fold. Trade-off: pipelines add a little ceremony, but the safety is worth it.

## 3. Explain k-fold cross-validation.
K-fold splits the training rows into k folds and lets each fold be the validation set once,
so every row is validated exactly once and I report the mean and spread. It gives a far more
stable estimate than a single split, whose score swings with luck. I coded it from scratch on
Day 3 and matched scikit-learn's fold sizes ([21,21,21,20,20] for 103 rows, k=5). Trade-off:
k fits cost k times as much, so on big data or slow models I use 3 to 5 folds.

## 4. When do you use stratified k-fold?
For classification, especially imbalanced, so every fold keeps the overall class mix. Plain
k-fold on sorted or rare-positive labels can leave a fold with no positives, making its recall
undefined. I coded stratified k-fold on Day 3 and showed plain k-fold leaving four of five
folds with zero positives on sorted labels. Trade-off: for grouped rows I use GroupKFold and
for time series TimeSeriesSplit instead, because the structure, not just the label, decides
the splitter.

## 5. Precision vs recall: define both and when you optimize each.
Precision is of everything I flagged, how much was right; recall is of everything truly
positive, how much I caught. They trade off through the threshold. I optimize recall for
cancer screening (missing a case is worst) and precision for spam (junking real mail is
worst). On Day 4 I coded both from scratch with zero-division handling and matched scikit-
learn. Trade-off: F1 balances them but hides which one is weak, so I report all three.

## 6. Why is accuracy misleading on imbalanced data?
Because a model can score high by always predicting the majority and learning nothing about
the rare class. On Day 4 a plain model on a 10-percent-positive set scored 0.926 accuracy but
only 0.333 recall. I judge imbalanced problems on recall, precision, and PR-AUC instead.
Trade-off: on balanced data accuracy is fine and easier to explain to stakeholders, so I do
not abandon it, I just match the metric to the balance.

## 7. ROC-AUC vs PR-AUC.
Both sweep every threshold, but ROC-AUC uses the false-positive rate, which stays tiny when
true negatives dominate, so it flatters a model on rare-positive data. PR-AUC ignores true
negatives and drops honestly. On balanced breast-cancer data I measured them nearly equal
(0.995 vs 0.994); the gap only opens under imbalance. So for fraud or churn I lead with
PR-AUC and recall. Trade-off: ROC-AUC is fine and more familiar on balanced problems.

## 8. How do you set a classification threshold?
Not at the default 0.5, but from the goal: I sweep thresholds on predict_proba and pick the
one that meets the business target. On Day 4 I wrote threshold_for_recall, which returns the
highest threshold still hitting a target recall, so it meets the goal at the best available
precision. For a 0.98-recall target it found a threshold near 0.09 at precision 0.75. Trade-
off: lower thresholds buy recall with precision, so the choice is a conversation about the
cost of each mistake.

## 9. How do you handle class imbalance?
Two independent levers: class_weight='balanced' makes rare-class mistakes count more (it only
touches the loss, so no leakage), and resampling changes the data (SMOTE, undersampling). On
Day 4, balancing lifted recall from 0.333 to 0.755, and a tuned threshold pushed it to 0.90.
Trade-off: resampling must happen inside the training fold only, or copies leak across the
split; class_weight avoids that trap, so I reach for it first.

## 10. Explain the bias-variance trade-off.
Bias is error from a model too simple to capture the pattern (underfitting); variance is error
from a model so flexible it fits noise (overfitting). Adding complexity lowers bias but raises
variance, so the goal is the middle. On Day 5 my validation curve over tree depth showed it:
validation peaked at depth 4, then training climbed to 1.0 while validation fell, the gap
widening. Trade-off: more data lets you afford more complexity before variance bites.

## 11. Name three fixes for an overfitting model.
Get more data, reduce complexity or add regularization, and use cross-validation to choose
settings honestly. On Day 5 I read the sweet spot straight off a validation curve rather than
trusting one split. Regularization is my usual first move on a linear model. Trade-off: too
much of any fix swings you into underfitting, so I confirm the change on cross-validation, not
the training score.

## 12. How do L1 and L2 regularization differ?
Both penalize large weights to curb overfitting; L2 shrinks them smoothly, L1 can drive some
to exactly zero, which is feature selection for free. On Day 5, L1 at a strong penalty kept 3
of 30 coefficients while L2 kept all 30. So I use L1 when I suspect many features are noise.
Trade-off: with correlated features L1 picks one arbitrarily, while L2 shares the weight, so
L2 or elastic net can be more stable.

## 13. Grid search vs randomized search, and the golden rule.
Grid search tries every combination (exhaustive but explodes in cost); randomized search
samples a fixed number and usually finds a near-best setting far cheaper, and can sample
continuous ranges. On Day 5 both hit CV F1 0.985 but random got there faster. The golden rule:
judge candidates on cross-validation over training data and confirm on the sealed test set
once (I got 0.979). Trade-off: with only two or three small-range parameters, grid is simplest.

## 14. How do random forests and gradient boosting differ?
A forest trains many independent trees on random subsets and averages them to cut variance;
boosting builds trees in sequence, each correcting the ensemble's errors, to cut bias. On
Day 6 gradient boosting topped my churn comparison (PR-AUC about 0.64) over logistic
regression. Forests are robust with little tuning; boosting is usually more accurate but can
overfit. Trade-off: in a regulated setting I might still pick logistic regression for
explainability.

## 15. Walk through a supervised project end to end.
Get and clean the data, do brief EDA, split honestly, build a leak-free preprocessing-plus-
model Pipeline, tune with cross-validation, and evaluate once on a sealed test set with the
metric that fits the goal. On Day 6 I did exactly this for churn: cleaned the Telco data,
chose PR-AUC, compared a dummy against logistic regression and gradient boosting, tuned the
best, and did error analysis by segment. Trade-off: the last mile (serving, monitoring,
retraining) is its own work, which weeks 3 and 4 cover.
