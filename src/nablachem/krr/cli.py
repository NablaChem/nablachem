import click
import importlib
import json
import os
import hashlib

from .utils import info, error, warning
from .dataset import DataSet
from .krr import AutoKRR
from . import features
from . import kernels


# Generate dynamic docstring with available representations
available_representations = features.list_available()
available_kernels = kernels.list_available()
MAIN_DOCSTRING = f"""Train KRR models on molecular data.

JSONL_PATH: Path to gzipped JSONL file containing molecular data
COLUMN_NAME: Property expression to predict using pandas DataFrame.eval() syntax.
            Can be a simple column name like 'energy' or a calculated expression
            like 'energy - baseline' or 'E_high - E_low'. For column names with
            special characters (dashes, spaces), use backticks like `E-high` - `E-low`.
REPRESENTATION_NAME: Name of the molecular representation to use.
                 Built-in representations: {', '.join(available_representations)}
                 Custom representations can be loaded from any importable module using
                 dotted notation, e.g. 'mymodule.MyRepresenter' or 'pkg.sub.MyRep'.
                 The class must implement the BaseRepresenter interface (compute/build).
KERNEL_NAME: Name of the kernel function to use.
         Available kernels: {', '.join(available_kernels)}

The dataset is split with the first maxcount molecules used for training,
and the remaining molecules used as holdout/test data.
"""


@click.command(help=MAIN_DOCSTRING)
@click.argument("jsonl_path", type=click.Path(exists=True))
@click.argument("column_name")
@click.argument("representation_name")
@click.argument("kernel_name")
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Maximum number of molecules to load (includes training + holdout). Defaults to maxcount + 2000",
)
@click.option("--mincount", default=128, help="Minimum training size")
@click.option(
    "--maxcount", default=2048, help="Maximum training size (rest used as holdout)"
)
@click.option(
    "--select",
    default=None,
    help="Selection expression for filtering dataset rows",
)
@click.option(
    "--detrend-atomic/--no-detrend-atomic",
    default=True,
    help="Enable/disable atomic count detrending (default: enabled)",
)
@click.option(
    "--detrend-pairs",
    default=None,
    type=str,
    help="Pairwise detrending functional form label (e.g. 'gCP'). Disabled by default.",
)
@click.option(
    "--holdout-residuals",
    default=None,
    help="Output JSONL file path for holdout residuals",
)
@click.option(
    "--elemental/--no-elemental",
    default=False,
    help="Mask cross-element atom pairs in local kernel (requires local representation)",
)
@click.option(
    "--alchemical",
    default=None,
    type=click.Path(exists=True),
    help="JSON file with per-element-pair weights {\"Z1,Z2\": float} (Z1<=Z2). Requires local representation.",
)
@click.option(
    "--archive", default="archive.json", help="Output file for KRR archive data"
)
@click.option(
    "--seed",
    default=-1,
    type=int,
    help="Random seed for numpy. Use -1 (default) for non-deterministic runs, or a non-negative integer for reproducible shuffles.",
)
def main(
    jsonl_path,
    column_name,
    representation_name,
    kernel_name,
    limit,
    mincount,
    maxcount,
    select,
    detrend_atomic,
    detrend_pairs,
    elemental,
    alchemical,
    holdout_residuals,
    archive,
    seed,
):
    if os.path.exists(archive):
        warning(f"Archive file {archive} will be overwritten.")

    # Set default limit if not specified
    if limit is None:
        limit = maxcount + 2000

    # Compute SHA256 hash of the input file for logging
    with open(jsonl_path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256")
    hash = digest.hexdigest()
    info("Starting", jsonl_path=jsonl_path, file_hash=hash)

    ds = DataSet(
        jsonl_path,
        column_name,
        limit=limit,
        select=select,
        seed=seed,
    )

    # Get the representation class dynamically
    if "." in representation_name:
        module_path, class_name = representation_name.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_path)
        except ImportError as e:
            error("Cannot import representation module", module=module_path, reason=str(e))
        try:
            rep_class = getattr(mod, class_name)
        except AttributeError:
            error("Class not found in module", module=module_path, class_name=class_name)
        rep = rep_class()
    else:
        rep_class_map = {}
        for name in available_representations:
            rep_class_map[name] = getattr(features, name)

        if representation_name not in rep_class_map:
            error(
                "Unknown representation",
                requested=representation_name,
                available=available_representations,
            )

        rep = rep_class_map[representation_name]()
    rep.build(ds)
    info("Prepared representation", first_entry_shape=ds.representations[0].shape)

    # Load alchemical weights if requested
    alchemical_weights = None
    if alchemical is not None:
        with open(alchemical) as f:
            raw = json.load(f)
        alchemical_weights = {}
        for key, val in raw.items():
            parts = key.split(",")
            if len(parts) != 2:
                error("Alchemical weight key must be 'Z1,Z2'", key=key)
            z1, z2 = int(parts[0]), int(parts[1])
            if z1 > z2:
                error("Alchemical weight keys must satisfy Z1 <= Z2", key=key, z1=z1, z2=z2)
            alchemical_weights[(z1, z2)] = abs(float(val))

    # Validate kernel name
    if kernel_name not in available_kernels:
        error(
            "Unknown kernel",
            requested=kernel_name,
            available=available_kernels,
        )

    # Instantiate kernel by name
    kernel_cls_map = {name: getattr(kernels, name) for name in available_kernels}
    kernel_func = kernel_cls_map[kernel_name]()
    autokrr = AutoKRR(
        ds,
        mincount,
        maxcount,
        detrend_atomic=detrend_atomic,
        detrend_pairs=detrend_pairs,
        kernel_func=kernel_func,
        elemental=elemental,
        alchemical_weights=alchemical_weights,
        seed=seed,
    )
    metadata = {
        "representation": representation_name,
        "kernel": kernel_name,
        "detrend_atomic": detrend_atomic,
        "detrend_pairs": detrend_pairs,
        "elemental": elemental,
        "alchemical": alchemical,
        "file_hash": hash,
        "file_path": jsonl_path,
        "column_name": column_name,
        "limit": limit,
        "select": select,
        "seed": seed,
    }
    autokrr.store_archive(archive, metadata)

    # Print learning curve table
    print("\nLearning Curve Results:")
    print("-" * 100)
    print(
        f"{'ntrain':>7} {'val_rmse':>10} {'test_rmse':>11} {'val_mae':>10} {'test_mae':>11} {'sigma':>12} {'lambda':>12}"
    )
    print("-" * 100)

    for ntrain in sorted(autokrr.results.keys()):
        result = autokrr.results[ntrain]
        if ntrain == 1:  # nullmodel
            print(
                f"{ntrain:>7} {result['val_rmse']:>10.4f} {result['test_rmse']:>11.4f} {result['val_mae']:>10.4f} {result['test_mae']:>11.4f} {'inf':>12} {'-':>12}"
            )
        else:
            params = result["parameters"]
            print(
                f"{ntrain:>7} {result['val_rmse']:>10.4f} {result['test_rmse']:>11.4f} {result['val_mae']:>10.4f} {result['test_mae']:>11.4f} {params['sigma']:>12.3e} {params['lambda']:>12.3e}"
            )
    print("-" * 100)

    # Generate holdout residuals JSONL if requested
    if holdout_residuals:
        ds.write_holdout_residuals_jsonl(
            autokrr.holdout_residuals, maxcount, holdout_residuals
        )
