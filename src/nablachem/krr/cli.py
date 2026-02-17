import click
import hashlib

from utils import info, error, result
from dataset import DataSet
from krr import AutoKRR
import features
import kernels


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
                 Available representations: {', '.join(available_representations)}
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
    "--holdout-residuals",
    default=None,
    help="Output JSONL file path for holdout residuals",
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
    holdout_residuals,
):
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
        representasssssstion_name=representation_name,
    )

    # Get the representation class dynamically
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
    rep.build([ds])
    info("Built representation", shape=ds.representations[0].shape)

    # Validate kernel name
    if kernel_name not in available_kernels:
        error(
            "Unknown kernel",
            requested=kernel_name,
            available=available_kernels,
        )

    execution_commands = {
        "jsonl file":jsonl_path,
        "Column analyzed":column_name,
        "Representation method":representation_name,
        "Kernel employed":kernel_name,
        "Run with limit":limit,
        "Minimum count considered":mincount,
        "Maximum count considered":maxcount,
        "Filters applied":select,
        "Atomic detrending":detrend_atomic,
        "Holdout":holdout_residuals,
    }




    # Instantiate kernel and get callable method
    k = kernels.Kernel()
    kernel_func = getattr(k, kernel_name)

    autokrr = AutoKRR(
        ds, mincount, maxcount, detrend_atomic=detrend_atomic, kernel_func=kernel_func, execution_commands=execution_commands
    )
    autokrr.store_archive("archive.json", execution_commands=execution_commands)

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


if __name__ == "__main__":
    main()
