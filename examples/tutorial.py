from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.event import DEFAULT_EVENTS, Events

# Installation
# pip install bayesian-optimization
# conda install -c conda-forge bayesian-optimization


def train_and_validate(lr, bs):
    """
    Internal 에 대해 알지 못하는 함수. (Training, validation 등)
    """
    return val_accuracy


def black_box_function(x, y, z):
    """
    일단 예시로 이 함수의 internal 을 모른다고 가정
    """
    return -x ** 2 - (y - 1) ** 2 + z

def get_discrete_idx(pbounds, discrete_names):
    param_names = list(pbounds.keys())
    param_names.sort()

    discrete_list = [0] * len(param_names)
    for i in range(len(param_names)):
        if param_names[i] in discrete_names:
            discrete_list[i] = 1

    return discrete_list

def optimize():
    """
    1. Basic 사용법
    """
    # Parameter 들의 범위를 지정해줌
    pbounds = {"x": (2, 4), "z": (0, 1), "y": (-3, 3)}
    discrete = ["y"]

    discrete_indices = get_discrete_idx(pbounds, discrete)

    optimizer = BayesianOptimization(
        f=black_box_function,  # Maximize 하고자 하는 함수
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,  # Optional, BO 에서 randomness 통제하기 위해 seed 입력 가능
        discrete=discrete_indices,
    )

    # init_points : 초기의 random exploration 개수
    # n_iter : 이후에 진행할 BO step 개수
    optimizer.maximize(init_points=2, n_iter=3)

    """
    2. 'maximize' 함수는 'Suggest-Evaluate(Probe)-Register' 반복하는 loop 의 wrapper
    """
    optimizer = BayesianOptimization(
        f=None, pbounds={"x": (-2, 2), "y": (-3, 3)}, verbose=2, random_state=1
    )

    # Exploration strategy. UCB (Upper Confidence Bound), EI (Expected Improvement) 등
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    # 1) Suggest
    next_point_to_probe = optimizer.suggest(utility)
    # 2) Evaluate
    target = black_box_function(**next_point_to_probe)
    # 3) Register
    optimizer.register(params=next_point_to_probe, target=target)

    """
    3. 추가로 가능한 것들
    """
    # (일부) Parameter 의 범위 변경
    optimizer.set_bounds(new_bounds={"x": (-2, 3)})

    # Evaluate 할 특정 point 지정해주기
    optimizer.probe(
        params={"x": 0.5, "y": 0.7},
        lazy=True,  # maximize 함수를 call 하는 시점에 evaluate 하겠다는 뜻
    )
    optimizer.maximize(init_points=0, n_iter=0)

    # Gaussian Process regressor 의 parameter 변경
    optimizer.set_gp_params(normalize_y=True)

    # BayesianOptimization object 가 일으키는 event 에 subscribe 및 listen 하는 Observer object 만들기
    # DEFAULT_EVENTS: optimization 시작, 각 step, 끝
    # event 가 일어날 때 불리는 callback 함수를 지정할 수 있음
    # 기본 제공되는 것으로는 optimization step 마다 json 으로 로깅하는 JSONLogger 등이 있음

    # Other patterns
    # parameter 범위는 log scale 로 잡고, optimizer 에 넘겨준 함수 f 내부에서 regular scale 로 변환해
    # 진짜로 optimize 하려는 함수를 다시 부르는 것. Optimizer 성능 향상에 좋다고 함 (?)
    # 비슷하게 discrete 쉽게 구현하려면 optimizer 에 넘겨준 함수 f 내부에서 적절히 parameter 를 discretize 한 후
    # evaluate 하려는 함수를 call 함


if __name__ == "__main__":
    optimize()
