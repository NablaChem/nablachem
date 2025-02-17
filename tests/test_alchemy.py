from nablachem.alchemy import MultiTaylor, Anygrad
import pandas as pd
import numpy as np


def test_1d():
    for i in range(100):
        np.random.seed(i)
        polynomial = np.poly1d(np.random.random(5))
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


def test_numerical_instability():
    ABs = np.array(
        [
            [0.0, 0.0],
            [-0.001, 0.0],
            [0.0, -0.001],
            [0.001, 0.0],
            [0.0, 0.001],
            [-0.001, -0.001],
            [-0.001, 0.001],
            [0.001, -0.001],
            [0.001, 0.001],
            [-0.002, 0.0],
            [0.0, -0.002],
            [0.002, 0.0],
            [0.0, 0.002],
            [-0.002, 0.002],
            [0.002, -0.002],
            [0.002, 0.002],
        ]
    )
    polynomial = lambda A, B: A * B * B * B + B * B + A
    A, B = 4, 5
    As = ABs[:, 0] + A
    Bs = ABs[:, 1] + B
    ys = polynomial(As, Bs)
    df = pd.DataFrame({"A": As, "B": Bs, "y": ys})
    mt = MultiTaylor(df, outputs=["y"])
    mt.reset_center(A=4, B=5)
    mt.build_model(3)


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


def test_analytical_gradients():
    import pyscf.gto
    import pyscf.scf

    atomspec = "C 0 0 0; O 0 0 1.1"
    basis = "sto-3g"

    mf = pyscf.scf.RHF(pyscf.gto.M(atom=atomspec, basis=basis, symmetry=False))
    mf.kernel()

    from nablachem.analyticgrads.AP_class import APDFT_perturbator as AP

    ap_nn = AP(mf, sites=[0, 1])
    val1 = 0.02656966379363701
    assert abs(ap_nn.af(0)[0, 2] - val1) < 1e-8
    assert abs(ap_nn.af(0)[1, 2] + val1) < 1e-8
    val2 = 0.27339079067037564
    assert abs(ap_nn.af(1)[0, 2] - val2) < 1e-8
    assert abs(ap_nn.af(1)[1, 2] + val2) < 1e-8

    ap_nn.build_all()


def test_anygrad_HF():
    import pyscf.gto
    import pyscf.scf

    atomspec = "C 0 0 0; O 0 0 1.1"
    basis = "sto-3g"

    mf = pyscf.scf.RHF(pyscf.gto.M(atom=atomspec, basis=basis, symmetry=False))
    mf.kernel()

    ag = Anygrad(mf, Anygrad.Property.ENERGY)
    grad = ag.get(Anygrad.Variable.POSNUC, method=Anygrad.Method.FINITE_DIFFERENCES)
    hess = ag.get(
        Anygrad.Variable.POSNUC,
        Anygrad.Variable.POSNUC,
        method=Anygrad.Method.FINITE_DIFFERENCES,
    )
    print(grad, hess)
    assert False
