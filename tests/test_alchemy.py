from nablachem.alchemy import MultiTaylor
import pandas as pd
import numpy as np


def test_1d():
    for i in range(100):
        np.random.seed(i)
        polynomial = np.poly1d(np.random.random(5))
        print(polynomial)
        x = 4
        dx = 0.01
        xs = (np.arange(5) - 2) * dx + x
        ys = polynomial(xs)
        df = pd.DataFrame({"x": xs, "y": ys})
        mt = MultiTaylor(df, outputs=["y"])
        mt.reset_center(x=4)
        mt.build_model(4)
        assert abs(mt.query(x=4)["y"] - polynomial(4)) < 1e-8
        assert abs(mt.query(x=5)["y"] - polynomial(5)) < 1e-4


def test_redundant_data():
    polynomial = lambda A, B: A
    A, B = 4, 5
    dx = 0.1
    As = (np.arange(2) - 1) * dx + A
    Bs = (np.arange(100) - 51) * dx / 10 + B
    As, Bs = np.meshgrid(As, Bs)
    As = As.flatten()
    Bs = Bs.flatten()
    ys = polynomial(As, Bs)
    df = pd.DataFrame({"A": As, "B": Bs, "y": ys})
    mt = MultiTaylor(df, outputs=["y"])
    mt.reset_center(A=4, B=5)
    mt.build_model(1)


def test_additional_emtpy_dimensions():
    polynomial = lambda A, B: A * B
    A, B = 4, 5
    dx = 0.1
    As = (np.arange(2) - 1) * dx + A
    Bs = (np.arange(5) - 2) * dx + B
    As, Bs = np.meshgrid(As, Bs)
    As = As.flatten()
    Bs = Bs.flatten()
    ys = polynomial(As, Bs)
    data = {}
    extras = {}
    for i in range(50):
        colname = f"C{i}"
        data[colname] = As * 0
        extras[colname] = 0
    # to be placed at the right of the df
    data.update({"A": As, "B": Bs, "y": ys})
    df = pd.DataFrame(data)
    mt = MultiTaylor(df, outputs=["y"])
    mt.reset_center(A=4, B=5, **extras)
    mt.build_model(0, [("A",), ("B",), ("A", "B")])


def test_2d():
    polynomial = lambda A, B: A * B * B * B + B * B + A
    A, B = 4, 5
    dx = 0.1
    As = (np.arange(5) - 2) * dx + A
    Bs = (np.arange(5) - 2) * dx + B
    As, Bs = np.meshgrid(As, Bs)
    As = As.flatten()
    Bs = Bs.flatten()
    ys = polynomial(As, Bs)
    df = pd.DataFrame({"A": As, "B": Bs, "y": ys})
    mt = MultiTaylor(df, outputs=["y"])
    mt.reset_center(A=4, B=5)
    mt.build_model(4)
    assert abs(mt.query(A=4, B=5)["y"] - polynomial(4, 5)) < 1e-8
    assert abs(mt.query(A=5, B=6)["y"] - polynomial(5, 6)) < 1e-4


def test_3d():
    polynomial = lambda A, B, C: A * B * C + B * B * C + A * C
    A, B, C = 4, 5, 6
    dx = 0.1
    As = (np.arange(5) - 2) * dx + A
    Bs = (np.arange(5) - 2) * dx + B
    Cs = (np.arange(5) - 2) * dx + C
    As, Bs, Cs = np.meshgrid(As, Bs, Cs)
    As = As.flatten()
    Bs = Bs.flatten()
    Cs = Cs.flatten()
    ys = polynomial(As, Bs, Cs)
    df = pd.DataFrame({"A": As, "B": Bs, "C": Cs, "y": ys})
    mt = MultiTaylor(df, outputs=["y"])
    mt.reset_center(A=A, B=B, C=C)
    mt.build_model(3)
    assert (
        abs(mt.query(A=A + 1, B=B + 1, C=C + 1)["y"] - polynomial(A + 1, B + 1, C + 1))
        < 1e-4
    )
