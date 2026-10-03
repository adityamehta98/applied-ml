# Training decisions checklist

## Before the run
- Look at the data first: labels, leakage between train and validation, class balance.
- Put every setting in one config and fix the seed.
- Try to overfit 20 examples. If you cannot, there is a bug, not a tuning problem.

## Reading the curves
- Both fall, small gap: healthy. Keep it.
- Both high and flat: underfitting. Raise the learning rate, train longer, or use a bigger model.
- Training falls, validation rises: overfitting. Stop at the best epoch, add data, add weight decay, or shrink.
- Spikes or NaN: the learning rate is too high. Lower it by 10x.

## What to change, in order
1. Data and bugs. 2. Learning rate. 3. Model size and regularisation. 4. Anything exotic.
Change one thing per run and write the note before you start.

## Budget
- Out of memory: lower the batch size, then bf16, then gradient accumulation, then a smaller model.
- Rank sweeps by validation loss and record the time each run took.
- Free compute first (Kaggle, Colab). Pay only when you have measured the bottleneck.
