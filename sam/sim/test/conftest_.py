import pytest


def pytest_addoption(parser):
    parser.addoption("--debug-sim", action="store_true", default=False,
                     help="Emit debug print statements. Use with `-s`")
    parser.addoption("--ssname", action="store", help="Suitesparse name for the end-to-end test")


def pytest_configure(config):
    config.addinivalue_line("markers", "suitesparse: mark test as needing suitesparse dataset to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--ssname"):
        # --ssname given in cli: do not skip apps/ tests
        return
    skip_ss = pytest.mark.skip(reason="Need --ssname option to run")
    for item in items:
        if "suitesparse" in item.keywords:
            item.add_marker(skip_ss)


@pytest.fixture
def debug_sim(request):
    return request.config.getoption("--debug-sim")


@pytest.fixture
def ssname(request):
    return request.config.getoption("--ssname")


@pytest.fixture
def samBench(benchmark):
    def f(func, extra_info = None, save_ret_val = False):
        # Take statistics based on 10 rounds.
        if extra_info is not None:
            for k, v in extra_info.items():
                benchmark.extra_info[k] = v
        if save_ret_val:
            benchmark.extra_info["return"] = func()
        benchmark.pedantic(func, rounds=1, iterations=1, warmup_rounds=0)                           
    return f