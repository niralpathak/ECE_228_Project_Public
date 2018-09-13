# Setup Prerequisite

## Kaggle Token
Generate your own valid Kaggle Token and place it under `src/data/kaggle.json`. 

To use the Kaggle API, sign up for a Kaggle account at `https://www.kaggle.com`. Then go to the 'Account' tab of your 
user profile (`https://www.kaggle.com/<username>/account`) and select 'Create API Token'. This will trigger the download 
of kaggle.json, a file containing your API credentials. 
  
The kaggle.json file is also required to have limited access rights, which can be set through the following:

```console
chmod 600 src/data/kaggle.json
```

# Initial Setup
```console
git clone https://github.com/niralpathak/ECE_228_Project.git
sh ECE_228_Project/src/data/setup.sh
```

The .sh script does the following:
1) Download, extract, and relocate train & test files from Kaggle
2) Install Python dependencies

# How to run code
In `src/model/keras/`, each main .py script corresponds to a particular network model .py script. For example, 
`main_custom_0.py` will train a network specified in `net_custom_0.py`.

The classes in `net_*.py` inherit from the Solver class in `solver.py` for modularity.

```console
cd ECE_228_Project/src/models/keras
python main_*.py --batch=<batch_size> --epochs=<epoch_size> --training_imgs=../../data/train-jpg/ --training_labels=../../data/train_v2.csv --patience=<patience_size> --testing_imgs=../../data/test-jpg/ --sample_submission=../../data/sample_submission_v2.csv --checkpoint_dir=<checkpoint_dir>
```

## Example

```console
python main_custom_0.py --batch=128 --epochs=10 --training_imgs=../../data/train-jpg/ --training_labels=../../data/train_v2.csv --patience=3 --testing_imgs=../../data/test-jpg/ --sample_submission=../../data/sample_submission_v2.csv --checkpoint_dir=./checkpoint_main
```
