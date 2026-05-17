def fit_standard_scaler(X):
    n_features = len(X[0])

    means = []
    stds = []

    for j in range(n_features):
        column = [row[j] for row in X]

        mean = sum(column) / len(column)
        variance = sum((v - mean) ** 2 for v in column) / len(column)
        std = variance ** 0.5

        means.append(mean)
        stds.append(std if std != 0 else 1)

    return means, stds


def transform_standard_scaler(X, means, stds):
    X_scaled = []

    for row in X:
        scaled_row = []

        for value, mean, std in zip(row, means, stds):
            scaled_row.append((value - mean) / std)

        X_scaled.append(scaled_row)

    return X_scaled