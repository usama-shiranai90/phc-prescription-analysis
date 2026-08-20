"""Head-to-head benchmarking of PHC-RxGen against non-neural baselines.

This package only *reads* the artefacts other workstreams produce (the trained
checkpoint, the processed corpus, the engineered feature table). It never
retrains the neural model and never writes outside its own directory.
"""
