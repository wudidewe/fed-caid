# Contributing

Run `python main.py --check` and the CPU synthetic example before proposing a change. Keep numerical changes separate from presentation or packaging changes. A changed experiment setting must be recorded and evaluated as a new experiment.

Preserve the complete ordered feature and label maps, the original source row identities, and the TRAIN/CAL/VAL roles. Never replace fixed role identities with a new split merely to make another source version load.

Keep dataset files, Kaggle credentials, checkpoints, private predictions, generated outputs and local machine paths out of Git. Publish only code, tests, instructions, third party notices and the declared row identity metadata.

The public experiment uses the repository's Apache-2.0 license. Retain third party license and attribution files.
