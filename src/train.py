import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42

MODEL_PATH = os.path.join(os.curdir, "models")
MINMAX_MODEL_PATH = os.path.join(MODEL_PATH, "minmax.pkl")
FOREST_MODEL_PATH = os.path.join(MODEL_PATH, "forest.pkl")
BAGGING_MODEL_PATH = os.path.join(MODEL_PATH, "bagging.pkl")
ONEVSREST_MODEL_PATH = os.path.join(MODEL_PATH, "onevsrest.pkl")

TEST_SIZE = 0.3


def main() -> None:

    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)35
    data = pd.DataFrame(wine_quality.data.features)
    classes = wine_quality.data.targets.squeeze()

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)
    data = pd.DataFrame(scaled_data, columns=data.columns)

    with open(MINMAX_MODEL_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    balancer = SMOTE(k_neighbors=4, random_state=RANDOM_STATE)
    balanced_data, balanced_classes = balancer.fit_resample(data, classes)

    print(balanced_classes.value_counts())

    data_train, data_test, class_train, class_test = train_test_split(
        balanced_data, balanced_classes, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
    criterion = ["gini", "entropy"]
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=10, num=2)]
    max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=20)]
    max_features = ["sqrt", "log2"]

    forest_parameter_map = {
        "n_estimators": n_estimators,
        "criterion": criterion,
        "min_samples_split": min_samples_split,
        "max_depth": max_depth,
        "max_features": max_features,
    }

    bagging_knn_parameter_map = {
        "n_estimators": n_estimators,
        "max_samples": [0.5, 0.75, 1.0],
        "max_features": [0.5, 0.75, 1.0],
        "bootstrap": [True, False],
        "estimator__n_neighbors": [3, 5, 7, 9],
        "estimator__weights": ["uniform", "distance"],
        "estimator__metric": ["euclidean", "manhattan"],
    }

    one_vs_rest_parameter_map = {
        "estimator__C": [0.01, 0.1, 1.0, 10.0],
        "estimator__solver": ["lbfgs", "liblinear"],
        "estimator__max_iter": [1000, 2000],
    }

    models = [
        (
            "Random Forest",
            RandomForestClassifier(random_state=RANDOM_STATE),
            forest_parameter_map,
            FOREST_MODEL_PATH,
        ),
        (
            "Bagging KNN Classifier",
            BaggingClassifier(
                estimator=KNeighborsClassifier(),
                random_state=RANDOM_STATE,
            ),
            bagging_knn_parameter_map,
            BAGGING_MODEL_PATH,
        ),
        (
            "One-vs-Rest Logistic Regression",
            OneVsRestClassifier(
                LogisticRegression(random_state=RANDOM_STATE)
            ),
            one_vs_rest_parameter_map,
            ONEVSREST_MODEL_PATH,
        ),
    ]

    from pprint import pprint

    for model_name, model, parameter_map, model_path in models:

        print(f"Treinando modelo: {model_name}")

        hyper_parameters = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameter_map,
            n_iter=10,
            cv=3,
            n_jobs=1,
            verbose=1,
        )

        hyper_parameters.fit(data_train, class_train)

        print("Melhores parametros:")
        pprint(hyper_parameters.best_params_)

        model = hyper_parameters.best_estimator_

        model = model.fit(data_train, class_train)

        os.makedirs(MODEL_PATH, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        class_predict = model.predict(data_test)

        labels = np.sort(class_test.unique())
        cm = confusion_matrix(class_test, class_predict, labels=labels)

        accuracy = accuracy_score(class_test, class_predict)
        print(f"Acurácia global: {accuracy * 100:.2f}%")

        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        for label, accuracy in zip(labels, class_accuracy):
            print(f"Classe {label}: {accuracy * 100:.2f}%")

        f1_micro = f1_score(class_test, class_predict, average="micro")
        print(f"F1-score micro: {f1_micro * 100:.2f}%")

        f1_macro = f1_score(class_test, class_predict, average="macro")
        print(f"F1-score macro: {f1_macro * 100:.2f}%")

        f1_weighted = f1_score(class_test, class_predict, average="weighted")
        print(f"F1-score weighted: {f1_weighted * 100:.2f}%")

        ConfusionMatrixDisplay.from_estimator(model, data_test, class_test)
        plt.show()


if __name__ == "__main__":
    main()
