import matplotlib.pyplot as plt


def plot_3d_classes(X, y, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    qpa = [row[0] for row in X]
    pulso = [row[1] for row in X]
    respiracao = [row[2] for row in X]

    scatter = ax.scatter(
        qpa,
        pulso,
        respiracao,
        c=y
    )

    ax.set_xlabel("qPA")
    ax.set_ylabel("Pulso")
    ax.set_zlabel("Respiração")
    ax.set_title(title)

    fig.colorbar(scatter, ax=ax, label="Classe")

    plt.show()


def plot_real_vs_pred_regression(y_true, y_pred):
    plt.figure()

    plt.scatter(y_true, y_pred)

    min_value = min(min(y_true), min(y_pred))
    max_value = max(max(y_true), max(y_pred))

    plt.plot(
        [min_value, max_value],
        [min_value, max_value]
    )

    plt.xlabel("Gravidade real")
    plt.ylabel("Gravidade prevista")
    plt.title("Regressão: real vs previsto")

    plt.show()


def plot_confusion_matrix(matrix):
    plt.figure()

    plt.imshow(matrix)

    plt.title("Matriz de confusão")
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")

    n_classes = len(matrix)

    plt.xticks(range(n_classes), range(1, n_classes + 1))
    plt.yticks(range(n_classes), range(1, n_classes + 1))

    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, str(matrix[i][j]), ha="center", va="center")

    plt.colorbar()
    plt.show()

import matplotlib.pyplot as plt


def plot_decision_boundary_2d_slice(
    model,
    X_original,
    y,
    means,
    stds,
    fixed_respiration,
    title,
    resolution=100
):
    qpa_values = [row[0] for row in X_original]
    pulso_values = [row[1] for row in X_original]

    qpa_min, qpa_max = min(qpa_values), max(qpa_values)
    pulso_min, pulso_max = min(pulso_values), max(pulso_values)

    qpa_step = (qpa_max - qpa_min) / resolution
    pulso_step = (pulso_max - pulso_min) / resolution

    grid_qpa = []
    grid_pulso = []
    grid_pred = []

    for i in range(resolution):
        qpa = qpa_min + i * qpa_step

        for j in range(resolution):
            pulso = pulso_min + j * pulso_step

            x = [qpa, pulso, fixed_respiration]

            x_scaled = [
                (x[k] - means[k]) / stds[k]
                for k in range(3)
            ]

            pred = model.predict(x_scaled)

            grid_qpa.append(qpa)
            grid_pulso.append(pulso)
            grid_pred.append(pred)

    plt.figure()

    plt.scatter(
        grid_qpa,
        grid_pulso,
        c=grid_pred,
        alpha=0.15,
        s=8
    )

    plt.scatter(
        [row[0] for row in X_original],
        [row[1] for row in X_original],
        c=y,
        edgecolors="black",
        s=35
    )

    plt.xlabel("qPA")
    plt.ylabel("Pulso")
    plt.title(title + f" | respiração fixa = {fixed_respiration}")

    plt.colorbar(label="Classe prevista")
    plt.show()