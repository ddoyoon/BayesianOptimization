from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.event import DEFAULT_EVENTS, Events
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    d = x.astype(int)
    return (
        np.exp(-(d - 2) ** 2) + np.exp(-(d - 6) ** 2 / 10) + 1 / (d ** 2 + 1)
    )


def plot_bo(f, bo, pbounds):
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle("Gaussian Process and Utility Function", fontsize=12)

    lower, upper = pbounds["x"]
    x = np.linspace(lower, upper, 1000)

    # plt.figure(figsize=(16, 9))
    axs[0].plot(x, f(x))

    # input to predict: shape = (n_samples, n_features)
    mean, sigma = bo._gp.predict(x.reshape(-1, 1), return_std=True)

    axs[0].plot(x, mean)
    axs[0].fill_between(x, mean + sigma, mean - sigma, alpha=0.1)

    axs[0].scatter(
        bo.space.params.flatten(), bo.space.target, c="red", s=50, zorder=10
    )

    y = bo.util.utility(x.reshape(-1, 1), bo._gp, None)
    ymax = max(y)
    xpos = np.where(y == ymax)
    xmax = x[xpos][0]
    axs[1].plot(x, y)
    axs[1].plot(xmax, ymax, marker="v")

    plt.xticks(np.arange(lower, upper + 1, 1.0))
    plt.show()


def optimize():
    pbounds = {"x": (-10, 10)}
    discrete = ["x"]

    optimizer = BayesianOptimization(
        f=f,  # Maximize 하고자 하는 함수
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,  # Optional, BO 에서 randomness 통제하기 위해 seed 입력 가능
        strategy="proposed",
        discrete=discrete,
    )

    optimizer.maximize(init_points=1, n_iter=10)

    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    next_to_probe = optimizer.suggest(utility)
    print(f"Next Point to Probe = {next_to_probe}")

    plot_bo(f, optimizer, pbounds)


if __name__ == "__main__":
    optimize()
