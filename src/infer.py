import os
import pickle

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from ucimlrepo import fetch_ucirepo

COLUMNS = ['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol']

MODEL_PATH = os.path.join(os.curdir, "models")
MINMAX_MODEL_PATH = os.path.join(MODEL_PATH, "minmax.pkl")
FOREST_MODEL_PATH = os.path.join(MODEL_PATH, "forest.pkl")
BAGGING_MODEL_PATH = os.path.join(MODEL_PATH, "bagging.pkl")
ONEVSREST_MODEL_PATH = os.path.join(MODEL_PATH, "onevsrest.pkl")

def main():
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)35
    X = pd.DataFrame(wine_quality.data.features)
    Y = wine_quality.data.targets.squeeze()

    X_input = pd.DataFrame([[7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4]], columns=COLUMNS)

    with open(MINMAX_MODEL_PATH, "rb") as f:
        min_max_scaler = pickle.load(f)

    X_normalized = min_max_scaler.transform(X)
    X_input_normalized = min_max_scaler.transform(X_input)

    with open(FOREST_MODEL_PATH, 'rb') as f:
        forest = pickle.load(f)

    with open(BAGGING_MODEL_PATH, "rb") as f:
        bagging = pickle.load(f)

    with open(ONEVSREST_MODEL_PATH, "rb") as f:
        onevsrest = pickle.load(f)

    models = [("Random Forest", forest), ("Bagging", bagging), ("One Vs Rest", onevsrest)]

    for model_name, model in models:
        Y_predict = model.predict(X_normalized)

        labels = np.sort(Y.unique())
        cm = confusion_matrix(Y, Y_predict, labels=labels)

        accuracy = accuracy_score(Y, Y_predict)
        print(f"Acurácia global: {accuracy * 100:.2f}%")

        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        for label, accuracy in zip(labels, class_accuracy):
            print(f"Classe {label}: {accuracy * 100:.2f}%")

        f1_micro = f1_score(Y, Y_predict, average="micro")
        print(f"F1-score micro: {f1_micro * 100:.2f}%")

        f1_macro = f1_score(Y, Y_predict, average="macro")
        print(f"F1-score macro: {f1_macro * 100:.2f}%")

        f1_weighted = f1_score(Y, Y_predict, average="weighted")
        print(f"F1-score weighted: {f1_weighted * 100:.2f}%")

        ConfusionMatrixDisplay.from_estimator(model, X_normalized, Y)
        plt.show()

        predict = model.predict(X_input_normalized)
        print(f"Input: {X_input}")
        print(f"Random Forest: {predict[0]}")


if __name__ == '__main__':
    main()
