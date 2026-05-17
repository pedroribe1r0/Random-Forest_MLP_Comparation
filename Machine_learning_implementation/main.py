from scale_trasnform import fit_standard_scaler, transform_standard_scaler
from MLP import MLPRegressor, MLPClassifier, rnd
from metrics import (mae, mse, rmse, accuracy, precision_recall_f1)
from plots import (plot_decision_boundary_2d_slice, plot_3d_classes, plot_real_vs_pred_regression, plot_confusion_matrix)
import os

def load_dataset(path, regression: bool):
    X = []
    y = []

    with open(path, "r") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            values = [float(v.strip()) for v in line.split(",")]

            qpa = values[3]
            pulso = values[4]
            respiracao = values[5]
            gravidade = values[6]
            


            X.append([qpa, pulso, respiracao])
            if regression:
                y.append(gravidade)
            else:
                classification = values[7]
                y.append(classification)

    return X, y

def train_test_split(X, y, test_size=0.2, seed=42):
    rnd.seed(seed)

    data = list(zip(X, y))
    rnd.shuffle(data)

    split_index = int(len(data) * (1 - test_size))

    train_data = data[:split_index]
    test_data = data[split_index:]

    X_train = [item[0] for item in train_data]
    y_train = [item[1] for item in train_data]

    X_test = [item[0] for item in test_data]
    y_test = [item[1] for item in test_data]

    return X_train, X_test, y_train, y_test
    

def main():

    base_dir = os.path.dirname(__file__)

    path_regression = os.path.join(
        base_dir,
        "./data/01_treino_sinais_vitais_sem_label.txt"
    )

    # =========================
    # REGRESSÃO
    # =========================

    print("\n=========================")
    print("TREINANDO MLP REGRESSORA")
    print("=========================\n")

    X_reg, y_reg = load_dataset(path_regression, regression=True)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg,
        y_reg
    )

    means, stds = fit_standard_scaler(X_train_reg)

    X_train_reg = transform_standard_scaler(
        X_train_reg,
        means,
        stds
    )

    X_test_reg = transform_standard_scaler(
        X_test_reg,
        means,
        stds
    )

    regressor = MLPRegressor(
        n_inputs=3,
        n_hidden=8,
        learning_rate=0.001
    )

    regressor.train(
        X_train_reg,
        y_train_reg,
        epochs=1000
    )

    y_pred_reg = regressor.predict_batch(
        X_test_reg
    )

    print("\nPrimeiras previsões da regressão:\n")

    for real, pred in zip(y_test_reg[:10], y_pred_reg[:10]):
        print(f"real={real:.2f} | pred={pred:.2f}")

    print()
    print("MSE:", mse(y_test_reg, y_pred_reg))
    print("RMSE:", rmse(y_test_reg, y_pred_reg))
    print("MAE:", mae(y_test_reg, y_pred_reg))

    plot_real_vs_pred_regression(
        y_test_reg,
        y_pred_reg
    )

    # =========================
    # CLASSIFICAÇÃO
    # =========================

    print("\n============================")
    print("TREINANDO MLP CLASSIFICADORA")
    print("============================\n")
    path_classification = os.path.join(
        base_dir,
        "./data/02_treino_sinais_vitais_com_label.txt"
    )

    X_clf, y_clf = load_dataset(path_classification, regression=False)

    y_clf = [int(label) for label in y_clf]

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf,
        y_clf
    )

    # usa normalização própria para o conjunto da classificação
    means_clf, stds_clf = fit_standard_scaler(X_train_clf)

    X_train_clf = transform_standard_scaler(
        X_train_clf,
        means_clf,
        stds_clf
    )

    X_test_clf = transform_standard_scaler(
        X_test_clf,
        means_clf,
        stds_clf
    )

    classifier = MLPClassifier(
        n_inputs=3,
        n_hidden=8,
        n_classes=4,
        learning_rate=0.001
    )

    classifier.train(
        X_train_clf,
        y_train_clf,
        epochs=1000
    )

    y_pred_clf = classifier.predict_batch(
        X_test_clf
    )

    print("\nPrimeiras previsões da classificação:\n")

    for real, pred in zip(y_test_clf[:10], y_pred_clf[:10]):
        print(f"real={real} | pred={pred}")

    correct = 0

    for real, pred in zip(y_test_clf, y_pred_clf):
        if real == pred:
            correct += 1

    acc = accuracy(y_test_clf, y_pred_clf)

    classification_metrics = precision_recall_f1(
        y_test_clf,
        y_pred_clf,
        n_classes=4
    )

    print()
    print("Acurácia:", acc)
    print("Precision macro:", classification_metrics["macro_precision"])
    print("Recall macro:", classification_metrics["macro_recall"])
    print("F1-score macro:", classification_metrics["macro_f1"])

    print("\nPrecision por classe:")
    for i, value in enumerate(classification_metrics["precision_per_class"]):
        print(f"Classe {i + 1}: {value:.4f}")

    print("\nRecall por classe:")
    for i, value in enumerate(classification_metrics["recall_per_class"]):
        print(f"Classe {i + 1}: {value:.4f}")

    print("\nF1-score por classe:")
    for i, value in enumerate(classification_metrics["f1_per_class"]):
        print(f"Classe {i + 1}: {value:.4f}")

    print("\nMatriz de confusão:")
    for row in classification_metrics["confusion_matrix"]:
        print(row)

    plot_confusion_matrix(
        classification_metrics["confusion_matrix"]
    )

    plot_3d_classes(
        X_test_clf,
        y_test_clf,
        "Classes reais no conjunto de teste"
    )

    plot_3d_classes(
        X_test_clf,
        y_pred_clf,
        "Classes previstas pela MLP"
    )

if __name__ == '__main__':
    main()