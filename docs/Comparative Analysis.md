# Comparative Analysis

Nov 22, 2026 · @camila & @nicholas

## Performance

All models are evaluated on the same 200-comment test set (100 positive, 100 negative) and compared with the stratified DummyClassifier baseline. Neural results are the mean ± standard deviation over five seeds (777–781), and the Transformer extension (DistilBERT) is evaluated on the same benchmark.

| Model | Accuracy | Precision | Recall | F1 | Kappa ($\kappa$) |
| --- | --- | --- | --- | --- | --- |
| Baseline (stratified) | 0.500 | 0.500 | 0.510 | 0.505 | 0.000 |
| Dense | 0.648 ± 0.026 | 0.723 ± 0.037 | 0.496 ± 0.129 | 0.575 ± 0.082 | 0.296 ± 0.052 |
| Vanilla RNN | 0.558 ± 0.032 | 0.577 ± 0.045 | 0.506 ± 0.142 | 0.524 ± 0.065 | 0.116 ± 0.064 |
| LSTM | 0.625 ± 0.043 | 0.720 ± 0.073 | 0.422 ± 0.130 | 0.519 ± 0.091 | 0.250 ± 0.086 |
| BiLSTM (recurrent best) | 0.677 ± 0.039 | 0.695 ± 0.068 | 0.670 ± 0.084 | 0.674 ± 0.019 | 0.354 ± 0.078 |
| **DistilBERT (Transformer)** | **0.900** | **0.892** | **0.910** | **0.901** | **0.800** |

- **Against the baseline:** Among models trained from scratch, the BiLSTM is the strongest recurrent architecture (accuracy 0.677, F1 0.674, kappa 0.354), followed by Dense (0.648, 0.575, 0.296) and the LSTM (0.625, 0.519, 0.250). The vanilla RNN is only 5.8 accuracy points above chance (kappa 0.116) and one seed fell below it (0.495), demonstrating severe instability due to vanishing gradients.
- **The Transformer leap (DistilBERT):** DistilBERT decisively shatters the performance ceiling of all models trained from scratch, reaching **0.900 accuracy, 0.901 F1, and 0.800 Cohen's Kappa**. This represents an absolute gain of **+22.3 percentage points in accuracy and +0.446 in Kappa over the BiLSTM**, and +40.0 points over the baseline.
- **Why does DistilBERT outperform recurrent models by such a wide margin?**
  1. *Transfer learning vs. scratch training:* Dense, RNN, and LSTM models had to learn token embeddings from scratch using only 800 training comments (vocab size ~2,490 tokens). With such small data, recurrent weights easily overfit or fail to capture rich semantics. DistilBERT comes pre-trained on Wikipedia and BookCorpus (~3.3 billion tokens), possessing deep prior syntactic and affective knowledge.
  2. *Subword tokenization (WordPiece):* While recurrent models mapped unseen test words to `<unk>` (losing information), DistilBERT's WordPiece vocabulary (30,522 tokens) splits out-of-vocabulary words into informative sub-tokens (`superunbelievable` $\rightarrow$ `super`, `##un`, `##bel`, `##ie`, `##vable`), preventing out-of-vocabulary degradation.
- **Does sequence order matter?** For models trained from scratch, the effect was marginal: Dense (which ignores order via average pooling) matched the unidirectional LSTM (0.648 vs. 0.625). However, BiLSTM improved balance across classes. DistilBERT uses learned positional embeddings, allowing full permutation-sensitive understanding without sequential recurrence bottlenecks.
- **Error balance:** Dense and LSTM exhibit high precision (0.723 and 0.720) but poor recall (0.496 and 0.422), disproportionately predicting the negative class. BiLSTM achieves balance (0.695 precision / 0.670 recall). DistilBERT achieves near-perfect balance (0.892 precision / 0.910 recall) on the 200 test comments (TN: 89, FP: 11, FN: 9, TP: 91).

## Training Complexity

Complexity scales sharply from the memoryless Dense network to the heavy Transformer architecture.

| Model | Total Parameters | Trainable Layers / Mechanism | Time per Run (s) | Computation over a Sequence ($T$ tokens) |
| --- | --- | --- | --- | --- |
| Dense | 143,873 | Embeddings + 2 linear layers | 0.5s (CPU) | Parallel: embedding lookup, average, feedforward |
| Vanilla RNN | 144,929 | Embeddings + 1 RNN tanh layer + Linear | 3.1s (CPU) | $T$ sequential updates ($h_t = \tanh(W x_t + U h_{t-1})$) |
| LSTM | 154,337 | Embeddings + 1 LSTM (4 gates) + Linear | 2.9s (CPU) | $T$ sequential updates ($4\times$ work per step: $i, f, o, c$) |
| BiLSTM | 166,913 | Embeddings + 2 LSTMs (forward + backward) | 6.3s (CPU) | $2T$ sequential updates across both directions |
| **DistilBERT** | **66,955,010** | **6 Transformer Encoders (12 heads, $d=768$) + Head** | **~148s - 235s (CPU)** | **Parallel self-attention ($O(T^2 \cdot d)$), $O(1)$ path length** |

- **Parameter distribution:** In the neural models trained from scratch, the embedding matrix (2,215 words $\times$ 64 dims $\approx$ 141,760 params) represented 85–99% of total parameters, leaving only 2k–25k parameters in the recurrent control. In DistilBERT, the 6 Transformer encoder blocks account for ~66.3M parameters, providing immense expressive capacity.
- **Optimization and convergence:**
  - *Vanilla RNN:* Highly unstable; required gradient clipping at 1.0 and early stopping. Cross-validation chose $epoch=1$, indicating rapid overfitting.
  - *LSTM / BiLSTM:* Stable convergence with early stopping (patience 5, stopped at epoch 2–5). Overfitting prevented hidden sizes $>32$.
  - *DistilBERT:* Optimized with AdamW ($lr = 3\times 10^{-5}$, weight decay $0.01$, linear schedule). Over 3 full epochs (150 steps), training loss steadily decreased from 0.482 to 0.171 while evaluation loss dropped to 0.282, confirming smooth fine-tuning without catastrophic forgetting.
- **Resource trade-off:** DistilBERT takes $\sim 25\times$ the training time of BiLSTM and $\sim 300\times$ that of Dense on CPU. However, for inference on single comments, DistilBERT executes in under 25ms, making it fully viable for real-time social media processing.

## Interpretability

Interpretability varies inversely with recurrent state complexity, while attention offers a distinct structural window.

- **Dense:** Highly transparent. Because sentence representations are simply the unweighted average of word embeddings, each word contributes additively before the classification layer.
- **Vanilla RNN & LSTM:** Bounded hidden vector $h_t$ rewritten at each timestep. While gating in LSTM demonstrates when information is retained or erased, the distributed representation makes tracing a prediction to an exact phrase non-trivial.
- **BiLSTM:** Concatenates hidden states from both directions, obscuring the attribution of individual tokens.
- **DistilBERT (Self-Attention):**
  - Attention maps in each of the 12 heads allow observing how tokens interact. For instance, the `[CLS]` token assigns high attention weights to polarizing adjectives (*"amazing"*, *"unwatchable"*) and negation modifiers (*"not"*, *"never"*).
  - *Theoretical caution:* In line with Jain & Wallace (2019) (*"Attention is not Explanation"*), attention distributions reflect internal routing rather than faithful post-hoc explanations. Nonetheless, gradient-based saliency confirms that DistilBERT isolates sentiment markers significantly better than recurrent hidden states.

## Relation to Turing Machine Concepts

The progression from Dense to Transformers directly mirrors fundamental paradigms of computation and memory in Turing Machines.

| Concept | Dense NN | Vanilla RNN | LSTM | BiLSTM | DistilBERT (Transformer) |
| --- | --- | --- | --- | --- | --- |
| **Memory Architecture** | None (instantaneous aggregation) | Single hidden vector $h_t$ overwritten sequentially | Gated cell state $c_t$ with explicit write, erase, read | Dual cell states ($c_t^{\rightarrow}, c_t^{\leftarrow}$) | Complete sequence retained; content-addressable memory |
| **Tape Processing** | Order-free lookup | One-way tape sweep (left to right) | One-way tape sweep with selective overwrite | Two-way tape sweep (bidirectional) | Direct random-access addressing across all cells |
| **Path Length Between Tokens** | N/A | $O(T)$ sequential steps | $O(T)$ sequential steps | $O(T)$ sequential steps | **$O(1)$ direct connection** |
| **Closest TM Analogue** | Combinational circuit / ROM lookup | Finite State Automaton with moving head | Turing Machine with bounded, gated tape segment | Two-head Turing Machine sweeping both directions | **Random Access Stored-Program Machine (RAM / Content-Addressable Memory)** |

- **Memory and Gating as TM Operations:**
  - In an LSTM, the input gate ($i_t$), forget gate ($f_t$), and output gate ($o_t$) act as differentiable analogues of a Turing Machine tape operations: write to memory, erase memory, and read from memory.
  - However, because recurrent hidden vectors are bounded in $\mathbb{R}^d$ with finite floating-point precision, practical RNNs and LSTMs operate as large **Finite State Automata (FSA)** rather than true Turing Machines with infinite tapes.
- **Self-Attention as Content-Addressable Tape Addressing:**
  - Recurrent models are constrained by sequential tape reading: to relate word $x_1$ with word $x_T$, information must traverse $T-1$ intermediate transitions, causing informational bottlenecks.
  - The Transformer's scaled dot-product attention mechanism:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
    implements an associative content-addressable memory. The query $Q$ addresses keys $K$ anywhere on the tape in a single step ($O(1)$ distance), functionally equivalent to a Turing Machine equipped with a random-access indexing mechanism.
- **Computability vs. Inductive Bias:**
  - Siegelmann & Sontag (1995) proved that recurrent networks with arbitrary real weights can simulate Universal Turing Machines. Pérez et al. (2019) similarly demonstrated that Transformers with hard attention are Turing-complete.
  - In our practical sentiment task with short Reddit comments ($T \le 30$ in recurrent, $T \le 64$ in Transformer), the language problem is decidable and strictly finite. Therefore, the computational limitation is never Turing-computability, but **generalization capacity and sample efficiency**. DistilBERT's pre-trained inductive bias provides the necessary prior to solve informal language ambiguity that no scratch-trained recurrent model could match on 800 samples.

## Takeaways and Conclusions

1. **Hierarchy of Architectures:**
   $$\text{Baseline (0.500)} < \text{Vanilla RNN (0.558)} < \text{LSTM (0.625)} < \text{Dense (0.648)} \le \text{BiLSTM (0.677)} \ll \mathbf{\text{DistilBERT (0.900)}}$$
2. **The BiLSTM was the best model trained from scratch:** It achieved the lowest F1 variance (0.019) and balanced precision/recall, outperforming unidirectional recurrent networks.
3. **DistilBERT is the definitive solution:** Achieving a 0.900 Accuracy and 0.800 Cohen's Kappa, it proves that pre-trained attention mechanisms are essential for nuanced sentiment classification on informal, noisy Reddit comments.
4. **Turing Machine Insight:** Replacing sequential tape traversal with parallel content-addressable self-attention eliminates gradient degradation and enables superior contextual reasoning.
