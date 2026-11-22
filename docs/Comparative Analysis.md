# Comparative Analysis

Nov 22, 2026 · @camila

## Performance

All models are evaluated on the same 200-comment test set (100 positive, 100 negative) and compared with the stratified DummyClassifier baseline. Neural results are the mean ± standard deviation over five seeds (777–781), so a gap only counts if it exceeds that spread.

| Model | Accuracy | Precision | Recall | F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| Baseline (stratified) | 0.500 | 0.500 | 0.510 | 0.505 | 0.000 |
| Dense | 0.648 ± 0.026 | 0.723 ± 0.037 | 0.496 ± 0.129 | 0.575 ± 0.082 | 0.296 ± 0.052 |
| Vanilla RNN | 0.558 ± 0.032 | 0.577 ± 0.045 | 0.506 ± 0.142 | 0.524 ± 0.065 | 0.116 ± 0.064 |
| LSTM | 0.625 ± 0.043 | 0.720 ± 0.073 | 0.422 ± 0.130 | 0.519 ± 0.091 | 0.250 ± 0.086 |
| BiLSTM (extra) | 0.677 ± 0.039 | 0.695 ± 0.068 | 0.670 ± 0.084 | 0.674 ± 0.019 | 0.354 ± 0.078 |

- **Against the baseline:** the BiLSTM is the best model (accuracy 0.677, F1 0.674, kappa 0.354), followed by Dense (0.648, 0.575, 0.296) and the LSTM (0.625, 0.519, 0.250). The vanilla RNN is only 5.8 accuracy points above chance (kappa 0.116) and one seed fell below it (0.495), so it is not clearly better than the baseline.
- **Does order matter?** Not clearly. Dense, which ignores word order, matches the unidirectional LSTM (accuracy 0.648 vs. 0.625, a gap smaller than one standard deviation). The BiLSTM's accuracy advantage over Dense (0.029) is also within the seed spread; what sets the BiLSTM apart is balance and stability.
- **Error balance:** Dense and the LSTM have high precision (0.723 and 0.720) but low recall (0.496 and 0.422): they label many positive comments as negative (on average 50.4 and 57.8 out of 100). The BiLSTM is balanced (precision 0.695, recall 0.670) and has the lowest F1 spread (0.019).
- **Stopwords:** keeping them did not help. Accuracy dropped for the vanilla RNN (0.558 to 0.533) and the BiLSTM (0.677 to 0.629) and was about equal for the LSTM (0.625 to 0.617).

## Training complexity

Complexity grows from Dense to BiLSTM in parameters and sequential computation, but the shared embedding table dominates the parameter count: about 2,215 words × 64 dimensions = 141,760 parameters (the vocabulary of the final runs), or 85–99% of each model. The layers on top differ much more in relative terms. All four models use hidden = 32, the value chosen by the grid search.

| Model | Total parameters | Layers on top of embeddings | Time per run (s) | Computation over a 30-token input |
| --- | --- | --- | --- | --- |
| Dense | 143,873 | 2,113 | 0.5 | Parallel: average of embeddings, then two linear layers |
| Vanilla RNN | 144,929 | 3,169 | 3.1 | 30 sequential steps, one tanh update each |
| LSTM | 154,337 | 12,577 | 2.9 | 30 sequential steps, about 4× the work per step (three gates plus the candidate) |
| BiLSTM | 166,913 | 25,153 | 6.3 | Two LSTM passes (left-to-right and right-to-left), about 2× the LSTM |

Times are per training run on CPU and include early stopping, so they also depend on how many epochs each run lasted, not only on the architecture.

- **Optimization:** recurrent models are trained with backpropagation through time and are harder to optimize (vanishing or exploding gradients in the vanilla RNN). We used gradient clipping at 1.0 and early stopping with patience 5. The vanilla RNN was the least stable: its accuracy ranged from 0.495 to 0.580 across seeds.
- **Tuning cost:** each model had a grid of 18 configurations (hidden, batch size, learning rate) × 3 cross-validation folds = 54 trainings, plus 5 final runs. The BiLSTM grid took about 5 minutes on CPU.
- **Overfitting risk:** more capacity did not help. Every model chose the smallest hidden size tried (32), and the BiLSTM's cross-validated accuracy was 0.57–0.62 with 64 or 128 units versus 0.64 with 32. In the final runs the BiLSTM stopped at epoch 2–4, so it overfits after a few passes over 800 comments. We controlled this with dropout 0.5, early stopping and a validation split taken only from train; the test set was used once per seed.

## Interpretability

Interpretability decreases as the model gains memory: the more the prediction depends on a hidden state built over the sequence, the harder it is to say which word caused it. This is a qualitative argument; we did not run attribution experiments.

- **Dense:** the most transparent. The sentence vector is the plain average of word embeddings, so each word contributes additively before the nonlinearity. Embeddings can be inspected (nearest words) and each word's influence on the output can be estimated directly.
- **Vanilla RNN:** the decision depends on a single hidden vector rewritten at every step. We can see how the state changes word by word, but no individual unit has a clear meaning.
- **LSTM:** the gates (input, forget, output) show when the model keeps or discards information, for example after a negation. The state is still a distributed vector, so the explanation is partial.
- **BiLSTM:** the least direct. It concatenates the final states of two passes, so a word's effect cannot be traced to a single position.

In all recurrent models, gradient-based saliency over the embeddings is the practical way to highlight influential words. Attention weights in the Transformer extension offer a similar view, but they are not a faithful explanation by themselves (Jain & Wallace, 2019).

## Relation to Turing Machine concepts

A Turing Machine combines a finite control, a tape it can read, write and move along, and transition rules. Each architecture reproduces a different part of that picture.

| Concept | Dense | Vanilla RNN | LSTM | BiLSTM | Transformer (extension) |
| --- | --- | --- | --- | --- | --- |
| Memory | None: a fixed function of the averaged words | Hidden vector, overwritten at each step | Cell state with input, forget and output gates | Two cell states, one per direction | No recurrent state: the whole input is kept as context |
| Sequence processing | Reads all tokens at once, ignores order | One token at a time, left to right | One token at a time, left to right | Two sweeps, in both directions | All positions in parallel through attention |
| Closest TM analogy | A lookup with no tape | Finite control without a tape | Control plus a gated, bounded "tape" cell | Head that can sweep both ways | Random access to every input cell |

- **Memory:** the LSTM gates act like write, erase and read decisions over a memory of fixed size, which is consistent with the LSTM outperforming the vanilla RNN here (kappa 0.250 vs. 0.116); however, the memoryless Dense model (kappa 0.296) was not worse than the LSTM, so memory alone did not decide the result on such short comments. But this memory is bounded and has finite precision, so in practice each recurrent network behaves like a very large finite-state machine, not like a machine with an unbounded tape. Neural Turing Machines (Graves et al., 2014) add an external, differentiable memory with read and write heads; that is the closest neural analogue of a tape, and we did not implement it.
- **Sequence processing:** a Turing Machine consumes its tape cell by cell with a moving head. The RNN and LSTM do the same one-way, with the hidden state playing the role of the control state; the BiLSTM sweeps back, and attention replaces the sweep with direct access.
- **Computability:** in theory, recurrent networks with unbounded precision can simulate Turing Machines (Siegelmann & Sontag, 1995). Our task does not test that power. Inputs have at most 30 tokens, so the set of possible inputs is finite and every labeling of it is computable; the networks always halt after a fixed number of steps, unlike a Turing Machine, which may loop forever. The limit here is not computability but generalization from 800 examples and an ambiguous target (labels come from human raters who sometimes disagree), which leaves an irreducible error.

## Takeaways and limitations

- **Trade-off:** more memory did not translate directly into better results. The vanilla RNN was the weakest, and Dense was competitive with the LSTM at a fraction of the cost (0.5 s vs. 2.9 s per run). The BiLSTM gave the best and most balanced result (F1 0.674, kappa 0.354) at about 12× the training time of Dense (6.3 s vs. 0.5 s) and with the least direct interpretability.
- **Best choice here:** the BiLSTM if balanced precision and recall matter; Dense if cost and transparency matter, since its accuracy is within the seed spread of the BiLSTM.
- **Statistical caution:** with 200 test comments, the 95% confidence interval of an accuracy of 0.65–0.68 is about ±6.5 points. Only the gaps to the baseline of Dense (+0.148), the LSTM (+0.125) and the BiLSTM (+0.177) exceed it; the vanilla RNN (+0.058) and the differences among the neural models do not.
- **Scope:** one dataset, 1,000 balanced comments, English only, and a single train/test split; absolute performance (accuracy up to 0.68) is limited by the small training set of 800 comments. The Transformer extension is reported separately.
