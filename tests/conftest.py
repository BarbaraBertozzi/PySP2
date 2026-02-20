import pytest
import matplotlib.pyplot as plt

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")