# Cuisine_Classifier

### Installation:

➢ conda create -n yummlyenv python=3.6 anaconda ➢ source activate yummlyenv
➢ conda install pip
➢ pip install -r requirement.txt

### How to run:
➢ python3 yummly_cuisine_classification.py cuisine.unlabeled.json.gz
1. It will ask for your choice of method:
1 for logistic regression, or 2 for random forests, or 3 for Text CNN.
2. Based on the choice, it will load the selected model.
3. After it will perform data loading and prediction of cuisines.
4. At the end, it will generate a CSV on your local machine.