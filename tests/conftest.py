import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest


@pytest.fixture(scope="session")
def sim():
    from simulation.environments.workcell_sim import WorkcellSim
    s = WorkcellSim()
    yield s
    s.close()


@pytest.fixture(scope="session")
def env():
    from simulation.environments.sorting_env import VolvoSortingEnv
    e = VolvoSortingEnv()
    yield e
    e.close()
