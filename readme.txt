Readme

Pipeline:

1. Train Autoencoder on the dataset with train_ae.py

2. Train general classifier with train_adaptive.py by activating --clean

3. Run compute_A.py to compute AA and AAA and save them. AAA can be very large, find a working directory with more than 1T free space to save them.

4. Perform adversarial training with train_adaptive.py
