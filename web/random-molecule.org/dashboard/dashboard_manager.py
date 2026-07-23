import os
import glob
import json
import gzip
import hashlib
import uuid
import sys
from nablachem.krr import features, kernels

ARCHIVE_DIR = "Archieve"
COMBINATIONS_DB = "combinations_db.json"
RESULTS_DB = "results_db.json"
SEEDS = [1, 2, 3, 4, 5]

def get_file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def extract_properties(filepath):
    properties = []
    try:
        if filepath.endswith('.gz'):
            f = gzip.open(filepath, 'rt')
        else:
            f = open(filepath, 'r')
            
        first_line = f.readline()
        if first_line:
            data = json.loads(first_line)
            properties = [k for k in data.keys() if k != 'xyz']
        f.close()
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
    return properties

def main():
    if not os.path.exists(ARCHIVE_DIR):
        print(f"Directory {ARCHIVE_DIR} does not exist. Please create it and add your dataset files.", file=sys.stderr)
        sys.exit(1)

    # 1. Discover datasets and properties
    dataset_files = glob.glob(os.path.join(ARCHIVE_DIR, "*.jsonl")) + \
                    glob.glob(os.path.join(ARCHIVE_DIR, "*.jsonl.gz"))
    
    datasets = {}
    for df in dataset_files:
        basename = os.path.basename(df)
        dataset_name = basename.split('.jsonl')[0]
        
        # Create a subdirectory for the processed results
        dataset_dir = os.path.join(ARCHIVE_DIR, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        props = extract_properties(df)
        datasets[dataset_name] = {
            "file": df,
            "dir": dataset_dir,
            "properties": props
        }
    
    # 2. Get available kernels and representations
    available_reps = features.list_available()
    available_kernels = kernels.list_available()
    
    global_reps = [r for r in available_reps if "Local" not in r]
    local_reps = [r for r in available_reps if "Local" in r]

    # 3. Handle run_settings.json
    SETTINGS_FILE = "run_settings.json"
    old_settings = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                old_settings = json.load(f)
        except Exception as e:
            print(f"Error loading {SETTINGS_FILE}: {e}", file=sys.stderr)

    # Always put references at the top of the file
    settings = {
        "__reference_global_representations__": global_reps,
        "__reference_local_representations__": local_reps,
        "__reference_kernels__": available_kernels,
        "datasets": old_settings.get("datasets", {})
    }

    settings_changed = False
    
    # Check if reference lists changed
    if old_settings.get("__reference_global_representations__") != global_reps or \
       old_settings.get("__reference_local_representations__") != local_reps or \
       old_settings.get("__reference_kernels__") != available_kernels:
        settings_changed = True

    for ds_name, ds_info in datasets.items():
        if ds_name not in settings["datasets"]:
            settings["datasets"][ds_name] = {
                "run_labels": ds_info["properties"],
                "run_global_representations": global_reps,
                "run_local_representations": local_reps,
                "run_kernels": available_kernels
            }
            settings_changed = True
        else:
            # Upgrade old format to new format if needed
            ds_settings = settings["datasets"][ds_name]
            if "run_representations" in ds_settings:
                old_reps = ds_settings.pop("run_representations")
                ds_settings["run_global_representations"] = [r for r in old_reps if "Local" not in r]
                ds_settings["run_local_representations"] = [r for r in old_reps if "Local" in r]
                settings_changed = True

    # Always write out the file to enforce the key ordering
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)

    if not os.path.exists(SETTINGS_FILE) or settings_changed:
        print(f"Updated {SETTINGS_FILE}.")
        print("Please review and edit this file to control which combinations to generate.")
        print("Run the script again to generate the commands.")
        sys.exit(0)

    # 4. Generate all combinations based on settings
    all_combinations = []
    for ds_name, ds_info in datasets.items():
        ds_settings = settings["datasets"].get(ds_name, {})
        labels_to_run = ds_settings.get("run_labels", [])
        global_reps_to_run = ds_settings.get("run_global_representations", [])
        local_reps_to_run = ds_settings.get("run_local_representations", [])
        kernels_to_run = ds_settings.get("run_kernels", [])

        for prop in labels_to_run:
            if prop not in ds_info["properties"]: continue
            
            # Combine global and local into a tuple with a boolean flag for 'is_local'
            reps_to_process = [(r, False) for r in global_reps_to_run] + [(r, True) for r in local_reps_to_run]
            
            for rep, is_local in reps_to_process:
                for kernel in kernels_to_run:
                    for seed in SEEDS:
                        combo = {
                            "dataset": ds_name,
                            "property": prop,
                            "representation": rep,
                            "kernel": kernel,
                            "seed": seed,
                            "is_local": is_local
                        }
                        all_combinations.append(combo)
    
    with open(COMBINATIONS_DB, 'w') as f:
        json.dump(all_combinations, f, indent=2)

    # 4. Scan existing results and rename them to hash if needed
    completed_runs = []
    for ds_name, ds_info in datasets.items():
        result_files = glob.glob(os.path.join(ds_info["dir"], "*.json"))
        for rf in result_files:
            file_hash = get_file_hash(rf)
            expected_name = f"{file_hash}.json"
            expected_path = os.path.join(ds_info["dir"], expected_name)
            
            if rf != expected_path:
                try:
                    os.rename(rf, expected_path)
                    rf = expected_path
                except Exception as e:
                    print(f"Failed to rename {rf} to {expected_name}: {e}", file=sys.stderr)
                    continue
            
            # Read metadata
            try:
                with open(rf, 'r') as f:
                    data = json.load(f)
                    metadata = data.get("metadata", {})
                    
                    rep = metadata.get("representation")
                    kernel = metadata.get("kernel")
                    prop = metadata.get("column_name")
                    seed = metadata.get("seed")
                    
                    if rep and kernel and prop and seed is not None:
                        completed_runs.append({
                            "dataset": ds_name,
                            "property": prop,
                            "representation": rep,
                            "kernel": kernel,
                            "seed": seed,
                            "hash": file_hash,
                            "file": rf
                        })
            except Exception as e:
                print(f"Failed to read metadata from {rf}: {e}", file=sys.stderr)

    with open(RESULTS_DB, 'w') as f:
        json.dump(completed_runs, f, indent=2)

    # 5. Diff combinations vs completed runs
    # Create a set of tuples for fast lookup of completed runs
    completed_set = set()
    for run in completed_runs:
        combo_tuple = (run["dataset"], run["property"], run["representation"], run["kernel"], run["seed"])
        completed_set.add(combo_tuple)

    missing_commands = []
    for combo in all_combinations:
        combo_tuple = (combo["dataset"], combo["property"], combo["representation"], combo["kernel"], combo["seed"])
        if combo_tuple not in completed_set:
            dataset_file = datasets[combo["dataset"]]["file"]
            dataset_dir = datasets[combo["dataset"]]["dir"]
            
            # Generate a temporary file name for this run
            temp_filename = f"temp_run_{uuid.uuid4().hex[:8]}.json"
            temp_filepath = os.path.join(dataset_dir, temp_filename)
            
            # Add --no-detrend-atomic flag for local representations
            detrend_flag = "--no-detrend-atomic " if combo.get("is_local", False) else ""
            cmd = f"nc-krr {dataset_file} '{combo['property']}' {combo['representation']} {combo['kernel']} {detrend_flag}--seed {combo['seed']} --archive {temp_filepath}"
            missing_commands.append(cmd)

    # 6. Output missing commands
    output_file = "missing_commands.txt"
    if missing_commands:
        print(f"# Missing {len(missing_commands)} combinations. Saving commands to {output_file}...")
        with open(output_file, "w") as f:
            for cmd in missing_commands:
                f.write(cmd + "\n")
        print(f"# Successfully generated {output_file}")
    else:
        print("# All combinations are complete!")
        # If the file exists from a previous run but we're complete, clear it out.
        if os.path.exists(output_file):
            with open(output_file, "w") as f:
                f.write("")

if __name__ == "__main__":
    main()
