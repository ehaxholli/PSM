1. Running hg_paral.py trains block i of the 200 blocks in the time-varying parallel score-matching approach. The id of the block to train is given by args.step_no. There are 200 blocks by default, which can of course be changed.
The training can be performed in parallel. For each block i, once the training is finished, the model i is saved as modeli in models folder. If you re-run the script for that same block, modeli will be loaded if it exist in the models folder, and training will resume.

2. Once all blocks have been trained (which of course can be done in parallel using a cluster), then test NLL can be calculated by running hg_paral_lkh.py

3. Once all blocks have been trained (which of course can be done in parallel using a cluster), then the evolution of the likelihood and the generation of the data via the CNF approach can be visualized by running hg_paral_gencnf.py

4. Once all blocks have been trained (which of course can be done in parallel using a cluster), then the evolution of the likelihood and the generation of the data via the SDE approach can be visualized by running hg_paral_gensde.py
