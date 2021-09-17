## Environment
* Ubuntu 18.04
* Python 3.9.5
* torch 1.9.0
* torchvision 0.10.0

## Run the Experiment
1. Prepare the dataset.
```
python generate_rotation.py
```
2. Apply our algorithm to fix the data.
```
python revert_rotation.py --method METHOD --dataset-name DATASET --model-name MODEL
```
3. According the the results, evluate the effectiveness.
```
python evluate.py --method METHOD --dataset-name DATASET --model-name MODEL
```