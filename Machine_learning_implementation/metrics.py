def mse(y_true, y_pred):
    total = 0

    for real, pred in zip(y_true, y_pred):
        total += (real - pred) ** 2

    return total / len(y_true)


def rmse(y_true, y_pred):
    return mse(y_true, y_pred) ** 0.5


def mae(y_true, y_pred):
    total = 0

    for real, pred in zip(y_true, y_pred):
        total += abs(real - pred)

    return total / len(y_true)


def accuracy(y_true, y_pred):
    correct = 0

    for real, pred in zip(y_true, y_pred):
        if real == pred:
            correct += 1

    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, n_classes):
    matrix = [
        [0 for _ in range(n_classes)]
        for _ in range(n_classes)
    ]

    for real, pred in zip(y_true, y_pred):
        matrix[real - 1][pred - 1] += 1

    return matrix


def precision_recall_f1(y_true, y_pred, n_classes):
    matrix = confusion_matrix(y_true, y_pred, n_classes)

    precisions = []
    recalls = []
    f1_scores = []

    for c in range(n_classes):
        tp = matrix[c][c]

        fp = 0
        for i in range(n_classes):
            if i != c:
                fp += matrix[i][c]

        fn = 0
        for j in range(n_classes):
            if j != c:
                fn += matrix[c][j]

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) != 0
            else 0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_precision = sum(precisions) / n_classes
    macro_recall = sum(recalls) / n_classes
    macro_f1 = sum(f1_scores) / n_classes

    return {
        "precision_per_class": precisions,
        "recall_per_class": recalls,
        "f1_per_class": f1_scores,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": matrix
    }