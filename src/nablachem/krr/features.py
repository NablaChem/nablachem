import numpy as np
from . import dataset
import inspect


class BaseRepresenter:
    def _prepare(self, molecules: list) -> None:
        pass

    def compute(self, molecules: list) -> list:
        raise NotImplementedError

    def build(self, ds, compatible_to=None) -> None:
        other_mols = [m for other in (compatible_to or []) for m in other.molecules]
        self._molecules = ds.molecules
        self._prepare(ds.molecules + other_mols)
        ds.representations = self

    def __len__(self) -> int:
        return len(self._molecules)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.compute(self._molecules[key])
        return self.compute([self._molecules[key]])[0]


class _DF(BaseRepresenter):
    def __init__(self, local: bool, dep: str):
        self._local = local
        self._dep = dep

    def _prepare(self, molecules: list) -> None:
        self._pad = max(len(mol.get_atomic_numbers()) for mol in molecules)
        all_charges = [mol.get_atomic_numbers() for mol in molecules]
        self._keys = np.unique(np.concatenate(all_charges))
        self._asize = {
            key: max(int((mol == key).sum()) for mol in all_charges)
            for key in self._keys
        }

    def compute(self, molecules: list) -> list:
        if not hasattr(self, "_pad"):
            raise RuntimeError("call build() before accessing representations")

        if self._dep == "mbdf":
            from .deps import mbdf

            call = mbdf.generate_mbdf
        else:
            from .deps import cmbdf

            call = cmbdf.generate_mbdf

        mols_charges = [mol.get_atomic_numbers() for mol in molecules]
        mols_coords = [mol.get_positions() for mol in molecules]
        natoms = [len(c) for c in mols_charges]

        kwargs = dict(progress_bar=False, local=self._local, pad=self._pad)
        if self._dep == "mbdf":
            kwargs["normalized"] = False
        if not self._local:
            kwargs["asize"] = self._asize
            kwargs["keys"] = self._keys
        reps = call(mols_charges, mols_coords, **kwargs)

        if self._local:
            return [reps[i][: natoms[i], :] for i in range(len(molecules))]
        else:
            return list(reps)


class MBDFLocal(_DF):
    def __init__(self):
        super().__init__(local=True, dep="mbdf")


class MBDFGlobal(_DF):
    def __init__(self):
        super().__init__(local=False, dep="mbdf")


class cMBDFLocal(_DF):
    def __init__(self):
        super().__init__(local=True, dep="cmbdf")


class cMBDFGlobal(_DF):
    def __init__(self):
        super().__init__(local=False, dep="cmbdf")


class _SLATM(BaseRepresenter):
    def __init__(self, local: bool = False):
        self._local = local

    def _prepare(self, molecules: list) -> None:
        import qmllib.representations

        all_charges = [mol.get_atomic_numbers() for mol in molecules]
        self._mbtypes = qmllib.representations.get_slatm_mbtypes(all_charges)

    def compute(self, molecules: list) -> list:
        import qmllib.representations

        reps = []
        for mol in molecules:
            charges = mol.get_atomic_numbers()
            coords = mol.get_positions()
            rep = qmllib.representations.generate_slatm(
                nuclear_charges=charges,
                coordinates=coords,
                mbtypes=self._mbtypes,
                local=self._local,
            )
            natom = len(charges)
            if self._local:
                if isinstance(rep, list):
                    reps.append(np.array(rep[:natom]))
                else:
                    reps.append(rep[:natom, :])
            else:
                reps.append(rep)
        return reps


class SLATMLocal(_SLATM):
    def __init__(self):
        super().__init__(local=True)


class SLATMGlobal(_SLATM):
    def __init__(self):
        super().__init__(local=False)


class _MACE(BaseRepresenter):
    def __init__(self, local: bool):
        self._local = local
        self._model = None

    def _prepare(self, molecules: list) -> None:
        if self._model is None:
            import warnings
            import contextlib

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from mace.calculators import mace_mp

                with contextlib.redirect_stdout(None):
                    self._model = mace_mp(
                        model="medium", device="", default_dtype="float64"
                    )

    def compute(self, molecules: list) -> list:
        if self._local:
            return [self._model.get_descriptors(mol) for mol in molecules]
        else:
            return [
                np.sum(self._model.get_descriptors(mol), axis=0) for mol in molecules
            ]


class MACEGlobal(_MACE):
    def __init__(self):
        super().__init__(local=False)


class MACELocal(_MACE):
    def __init__(self):
        super().__init__(local=True)


class _FCHL19(BaseRepresenter):
    def __init__(self, local: bool):
        self._local = local

    def _prepare(self, molecules: list) -> None:
        self._all_element_numbers = sorted(
            set(n for mol in molecules for n in mol.get_atomic_numbers())
        )

    def compute(self, molecules: list) -> list:
        import qmllib.representations

        reps = []
        for mol in molecules:
            rep = qmllib.representations.generate_fchl19(
                mol.get_atomic_numbers(),
                mol.get_positions(),
                elements=self._all_element_numbers,
            )
            if not self._local:
                rep = np.sum(rep, axis=0)
            reps.append(rep)
        return reps


class FCHL19Global(_FCHL19):
    def __init__(self):
        super().__init__(local=False)


class FCHL19Local(_FCHL19):
    def __init__(self):
        super().__init__(local=True)


class _SOAP(BaseRepresenter):
    """SOAP representation (dscribe) with "mu2" compression.

    A single element-agnostic ``mu2`` channel yields a fixed-size feature vector
    independent of the number of species; element identity is reintroduced via
    ``species_weighting`` (each species weighted by ``Z ** weight_exponent``).

    The hyperparameters below were optimized to minimize VQM24 Etot validation
    RMSE at ntrain=1024 (~12, down from ~29 for the original sigma=1.0,
    unit-weighted descriptor). The key findings:

    * ``sigma`` (atomic density width) is the dominant lever. The dscribe
      default of 1.0 over-smears the density; ~0.12 is far sharper and roughly
      halves the error. Going below ~0.1 does not help further.
    * ``weight_exponent`` ~ 0.45 (i.e. weights ~ sqrt(Z)) is optimal. Unit
      weights (exponent 0) lose element identity (~21 RMSE); raw Z (exponent 1)
      has too large a dynamic range -- cross-terms scale with Z_i*Z_j (~1225x
      for Br-Br vs H-H) -- which both hurts accuracy and makes the Gaussian
      kernel ill-conditioned.
    * Per-feature standardization is deliberately NOT applied. It helps at the
      (bad) sigma=1.0 setting by taming the magnitude spread, but at the sharp
      sigma~0.12 optimum it upweights low-variance noise dimensions and makes
      the error notably worse (~16 vs ~12).
    * r_cut=6, n_max=8, l_max=6 were confirmed optimal; larger bases do not
      help and can reintroduce conditioning problems.
    """

    # SOAP hyperparameters (optimized; see class docstring)
    r_cut = 6.0
    n_max = 8
    l_max = 6
    sigma = 0.12
    # species weighting: w(Z) = Z ** weight_exponent
    weight_exponent = 0.45

    def __init__(self, local: bool):
        self._local = local
        self._soap = None

    def _prepare(self, molecules: list) -> None:
        from dscribe.descriptors import SOAP

        all_charges = [mol.get_atomic_numbers() for mol in molecules]
        species = sorted(int(z) for z in np.unique(np.concatenate(all_charges)))
        species_weighting = {z: float(z) ** self.weight_exponent for z in species}
        self._soap = SOAP(
            species=species,
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
            sigma=self.sigma,
            periodic=False,
            compression={"mode": "mu2", "species_weighting": species_weighting},
        )

    def compute(self, molecules: list) -> list:
        if self._soap is None:
            raise RuntimeError("call build() before accessing representations")
        reps = []
        for mol in molecules:
            rep = self._soap.create(mol)
            if not self._local:
                rep = np.sum(rep, axis=0)
            reps.append(rep)
        return reps


class SOAPLocal(_SOAP):
    def __init__(self):
        super().__init__(local=True)


class SOAPModLocal(_SOAP):
    """SOAPLocal retuned for VQM24 Etot at ntrain=2048 with alchemical kernel weights.

    Same single ``mu2`` density channel and basis (r_cut=6, n_max=8, l_max=6,
    all reconfirmed optimal -- both larger and smaller bases/cutoffs are worse),
    but two density settings differ from :class:`SOAPLocal`:

    * ``sigma`` 0.12 -> 0.18: at this larger training size the sharp 0.12 width
      over-resolves the density; a slightly broader gaussian generalizes better.
    * ``weight_exponent`` 0.45 -> 0.25: when an alchemical (element-pair) kernel
      already carries element identity, a flatter species weighting in the
      descriptor is less redundant and clearly better. The optimum is a broad,
      shallow basin around sigma~0.18-0.20, weight_exponent~0.20-0.25.

    Measured mean validation RMSE over seeds {4, 5} (alchemical=best_alch.json,
    --mincount 2048 --limit 2049): 8.84, down from 10.22 for SOAPLocal (-13.5%).
    """

    sigma = 0.18
    weight_exponent = 0.25

    def __init__(self):
        super().__init__(local=True)


class SOAPModGlobal(SOAPModLocal):
    """Global counterpart of :class:`SOAPModLocal`.

    Identical density settings and basis; the per-atom SOAP vectors are summed
    into a single molecular descriptor (``_SOAP.compute`` sums when not local).
    """

    def __init__(self):
        super().__init__()
        self._local = False


class _MessagePassing:
    """Mixin adding distance-weighted message passing to any local representer.

    For each atom *i* the base atomic representation is enriched with a
    distance-weighted, additive residual contribution from every other atom::

        r'_i = r_i + beta * sum_{j != i} exp(-d_ij / decay) * r_j

    repeated ``mp_hops`` times. The descriptor dimension is unchanged, so the
    result drops straight into the local kernels. The exponential decay
    suppresses far-away atoms smoothly (no hard cutoff), and ``beta -> 0`` or
    ``decay -> 0`` recovers the bare base representation.

    Hyperparameters are class attributes so each concrete representation can
    override them by subclassing.
    """

    # exponential decay length in Angstrom (larger -> wider neighbourhood)
    mp_decay = 2.5
    # residual mixing weight (0 recovers the base representation)
    mp_beta = 1.0
    # number of message-passing rounds (>1 risks over-smoothing)
    mp_hops = 1

    def compute(self, molecules: list) -> list:
        reps = super().compute(molecules)
        out = []
        for rep, mol in zip(reps, molecules):
            pos = mol.get_positions()
            if rep.shape[0] < 2:
                out.append(rep)
                continue
            diffs = pos[:, None, :] - pos[None, :, :]
            d = np.linalg.norm(diffs, axis=2)
            W = np.exp(-d / self.mp_decay)
            np.fill_diagonal(W, 0.0)
            for _ in range(self.mp_hops):
                rep = rep + self.mp_beta * (W @ rep)
            out.append(rep)
        return out


class MBDFLocalMP(_MessagePassing, MBDFLocal):
    mp_decay = 2.5
    mp_beta = 1.0
    mp_hops = 1


class cMBDFLocalMP(_MessagePassing, cMBDFLocal):
    mp_decay = 2.5
    mp_beta = 0.001
    mp_hops = 1


class SLATMLocalMP(_MessagePassing, SLATMLocal):
    mp_decay = 2.5
    mp_beta = 1.0
    mp_hops = 1


class MACELocalMP(_MessagePassing, MACELocal):
    mp_decay = 2.5
    mp_beta = 1.0
    mp_hops = 1


class FusedFCLocal(BaseRepresenter):
    """Local representation fusing FCHL19 with cMBDF channels.

    Each atom's (tuned) FCHL19 descriptor is augmented -- "message passed" --
    with a complementary cMBDF density descriptor. The two capture different
    physics (distance-resolved many-body terms vs. density-based channels), and
    fusing them markedly lowers the learning-curve error relative to either
    alone. Both blocks are rescaled to a common per-atom norm so the single
    length-scale Gaussian kernel weighs them comparably, and the cMBDF block is
    mixed in with relative weight ``cmbdf_weight``.

    Hyperparameters are class attributes so they can be overridden by
    subclassing.
    """

    # tuned FCHL19 representation hyperparameters (defaults are QM9-era values;
    # these wider radial/angular widths and lighter three-body weight suit the
    # heavier-element VQM24 chemistry)
    fchl_eta2 = 0.17
    fchl_eta3 = 1.5
    fchl_three_body_weight = 5.0
    # relative weight of the cMBDF block after per-block normalization
    cmbdf_weight = 1.6
    # per-atom vectors are summed into one molecular descriptor when False
    _local = True

    def _fchl(self, molecules: list) -> list:
        import qmllib.representations

        return [
            qmllib.representations.generate_fchl19(
                mol.get_atomic_numbers(),
                mol.get_positions(),
                elements=self._all_element_numbers,
                eta2=self.fchl_eta2,
                eta3=self.fchl_eta3,
                three_body_weight=self.fchl_three_body_weight,
            )
            for mol in molecules
        ]

    def _cmbdf(self, molecules: list) -> list:
        from .deps import cmbdf

        charges = [mol.get_atomic_numbers() for mol in molecules]
        coords = [mol.get_positions() for mol in molecules]
        natoms = [len(c) for c in charges]
        reps = cmbdf.generate_mbdf(
            charges, coords, local=True, pad=self._pad, progress_bar=False
        )
        return [reps[i][: natoms[i], :] for i in range(len(molecules))]

    def _prepare(self, molecules: list) -> None:
        self._all_element_numbers = sorted(
            set(n for mol in molecules for n in mol.get_atomic_numbers())
        )
        self._pad = max(len(mol.get_atomic_numbers()) for mol in molecules)
        fchl = self._fchl(molecules)
        cmbdf = self._cmbdf(molecules)
        Xf = np.concatenate([r for r in fchl if r.shape[0] > 0], axis=0)
        Xc = np.concatenate([r for r in cmbdf if r.shape[0] > 0], axis=0)
        # per-atom RMS norm of each block, used to put them on a common scale
        self._fchl_scale = np.sqrt((Xf**2).sum(axis=1).mean())
        self._cmbdf_scale = np.sqrt((Xc**2).sum(axis=1).mean())

    def compute(self, molecules: list) -> list:
        if not hasattr(self, "_fchl_scale"):
            raise RuntimeError("call build() before accessing representations")
        fchl = self._fchl(molecules)
        cmbdf = self._cmbdf(molecules)
        reps = [
            np.concatenate(
                [
                    rf / self._fchl_scale,
                    self.cmbdf_weight * rc / self._cmbdf_scale,
                ],
                axis=1,
            )
            for rf, rc in zip(fchl, cmbdf)
        ]
        if not self._local:
            reps = [np.sum(rep, axis=0) for rep in reps]
        return reps


class FusedFCGlobal(FusedFCLocal):
    """Global counterpart of :class:`FusedFCLocal`.

    Same fused FCHL19 + cMBDF per-atom descriptor and hyperparameters; the
    per-atom vectors are summed into a single molecular descriptor.
    """

    _local = False


# Representations defined in dedicated modules (registered as built-in names).
from .physpot import PhysPot  # noqa: E402,F401
from .protorep import ProtoRep  # noqa: E402,F401


def list_available():
    """Return string names of all BaseRepresenter subclasses that don't start with underscore."""
    current_module = inspect.getmembers(
        inspect.getmodule(inspect.currentframe()), inspect.isclass
    )

    available = []
    for name, cls in current_module:
        if (
            name != "BaseRepresenter"
            and issubclass(cls, BaseRepresenter)
            and not name.startswith("_")
        ):
            available.append(name)

    return available
