"""Multi-label reframing of PHC-RxGen.

The head-to-head benchmark (`bench/head_to_head.py`) compared an
*autoregressive set decoder* against one-vs-rest logistic regression and the
linear arms won by ~0.09 micro-F1. That comparison confounds two things:
the hypothesis class (neural vs linear) and the output parameterisation
(ordered sequence with a learned stop symbol vs. independent per-label
decisions with a tuned threshold).

This package removes the second confound. It keeps the PHC-RxGen encoder
(char-CNN + word BiLSTM, per-vital tokens, history GRU, transformer fusion)
exactly as built, deletes the decoder, and attaches a linear multi-label head
trained with BCEWithLogitsLoss and a threshold tuned on validation -- i.e. the
neural model is given the same decision rule the linear baselines get.
"""
