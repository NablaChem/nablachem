import os
import glob
import json
import gzip
import hashlib
import uuid
import sys
import re
import pandas as pd
from nablachem.krr import features, kernels
                                
ARCHIVE_DIR = "archive"
DATASETS_DIR = "datasets"
RESULTS_DATAFRAME = "results_dataframe.parquet"

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

def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)

def main():
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)
        print(f"Directory {DATASETS_DIR} created. Please add your dataset files there.", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)

    # 1. Discover datasets and properties
    dataset_files = glob.glob(os.path.join(DATASETS_DIR, "*.jsonl")) + \
                    glob.glob(os.path.join(DATASETS_DIR, "*.jsonl.gz"))
    
    datasets = {}
    for df in dataset_files:
        basename = os.path.basename(df)
        if basename.endswith('.jsonl.gz'):
            dataset_name = basename[:-9]
        elif basename.endswith('.jsonl'):
            dataset_name = basename[:-6]
        
        props = extract_properties(df)
        datasets[dataset_name] = {
            "file": df,
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

    default_exclusions = {
        "general": {
            "kernels": [],
            "properties": ["smiles", "ref number", "xyz"],
            "representations": []
        },
        "combinations": []
    }

    exclusions = old_settings.get("exclusions", default_exclusions)
    if "general" not in exclusions:
        exclusions["general"] = default_exclusions["general"]
    for key in ["kernels", "properties", "representations"]:
        if key not in exclusions["general"]:
            exclusions["general"][key] = default_exclusions["general"][key]
    if "combinations" not in exclusions:
        exclusions["combinations"] = default_exclusions["combinations"]

    settings = {
        "__reference_flags__": [
            "--limit",
            "--mincount",
            "--maxcount",
            "--select",
            "--detrending",
            "--holdout-residuals",
            "--elemental",
            "--no-elemental",
            "--alchemical",
            "--owl",
            "--archive",
            "--seed",
            "--predict"
        ],
        "__reference_global_representations__": global_reps,
        "__reference_local_representations__": local_reps,
        "__reference_kernels__": available_kernels,
        "exclusions": exclusions
    }

    settings_changed = False
    
    if old_settings.get("__reference_global_representations__") != global_reps or \
       old_settings.get("__reference_local_representations__") != local_reps or \
       old_settings.get("__reference_kernels__") != available_kernels or \
       "datasets" in old_settings:
        settings_changed = True

    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)

    if not os.path.exists(SETTINGS_FILE) or settings_changed:
        print(f"Updated {SETTINGS_FILE}.")
        print("Please review and edit this file to control which combinations to generate.")
        print("Run the script again to generate the commands.")
        sys.exit(0)

    # 4. Generate all combinations based on settings
    all_combinations = []
    
    gen_ex = settings["exclusions"]["general"]
    excluded_kernels = set(gen_ex.get("kernels", []))
    excluded_props = set(gen_ex.get("properties", []))
    excluded_reps = set(gen_ex.get("representations", []))
    combo_ex = settings["exclusions"].get("combinations", [])
    
    def is_combination_excluded(ds, prop, rep, kern):
        for ex in combo_ex:
            match = True
            if "dataset" in ex and ex["dataset"] != ds: match = False
            if "property" in ex and ex["property"] != prop: match = False
            if "representation" in ex and ex["representation"] != rep: match = False
            if "kernel" in ex and ex["kernel"] != kern: match = False
            if match:
                return True
        return False

    detrending_opts = ["atomic", "pairs", "charge", "spin", ""]
    owl_opts_global = [""]
    elemental_opts_local = [True, False]
    elemental_opts_global = [False]
    seed_opts = [1, 2, 3, 4, 5]

    for ds_name, ds_info in datasets.items():
        all_props = ds_info["properties"]
        for prop in all_props:
            if prop in excluded_props: continue
            
            prop_lower = prop.lower()
            current_owl_opts_local = [""]
            if re.search(r'homo|gap', prop_lower):
                current_owl_opts_local.append("HOMO")
            if re.search(r'lumo|gap', prop_lower):
                current_owl_opts_local.append("LUMO")
            
            reps_to_process = [(r, False) for r in global_reps] + [(r, True) for r in local_reps]
            
            for rep, is_local in reps_to_process:
                if rep in excluded_reps: continue
                for kernel in available_kernels:
                    if kernel in excluded_kernels: continue
                    if is_combination_excluded(ds_name, prop, rep, kernel): continue
                        
                    for det in detrending_opts:
                        for owl in (current_owl_opts_local if is_local else owl_opts_global):
                            for ele in (elemental_opts_local if is_local else elemental_opts_global):
                                for seed in seed_opts:
                                    combo = {
                                        "dataset": ds_name,
                                        "property": prop,
                                        "representation": rep,
                                        "kernel": kernel,
                                        "detrending": det,
                                        "owl": owl,
                                        "elemental": ele,
                                        "seed": seed,
                                        "is_local": is_local
                                    }
                                    all_combinations.append(combo)
    # 5. Build Pandas DataFrame and keep existing metadata
    data_rows = []
    result_files = glob.glob(os.path.join(ARCHIVE_DIR, "**", "*.json"), recursive=True)
    completed_runs = []
    
    for rf in result_files:
        try:
            with open(rf, 'r') as f:
                data = json.load(f)
                meta = data.get("metadata", {})
                lc = data.get("learning_curve", [])
                
                rep = meta.get("representation")
                kernel = meta.get("kernel")
                col_name = meta.get("column_name")
                dataset = meta.get("dataset")
                
                if "detrending" in meta:
                    det_list = meta["detrending"]
                    det = det_list[0] if det_list else ""
                else:
                    if meta.get("detrend_atomic"): det = "atomic"
                    elif meta.get("detrend_pairs"): det = "pairs"
                    else: det = "atomic"
                    
                owl = meta.get("owl", "")
                if owl is None: owl = ""
                
                ele = meta.get("elemental", False)
                seed = meta.get("seed", -1)
                
                file_hash = get_file_hash(rf)
                
                if rep and kernel and col_name and dataset:
                    completed_runs.append({
                        "dataset": dataset,
                        "property": col_name,
                        "representation": rep,
                        "kernel": kernel,
                        "detrending": det,
                        "owl": owl,
                        "elemental": ele,
                        "seed": seed,
                        "hash": file_hash,
                        "file": rf
                    })
                    
                    for step in lc:
                        ntrain = step.get('ntrain', 0)
                        if is_power_of_two(ntrain) and ntrain <= 20000:
                            data_rows.append({
                                'dataset': dataset,
                                'column_name': col_name,
                                'representation': rep,
                                'kernel': kernel,
                                'detrending': det,
                                'owl': owl,
                                'elemental': ele,
                                'seed': seed,
                                'ntrain': ntrain,
                                'test_mae': step.get('test_mae'),
                                'validation_mae': step.get('validation_mae', step.get('test_mae')),
                                'uid': meta.get('uid', str(uuid.uuid4())),
                                'hash': file_hash
                            })
        except Exception as e:
            print(f"Error parsing {rf}: {e}", file=sys.stderr)

    df = pd.DataFrame(data_rows)
    if not df.empty:
        df.to_parquet(RESULTS_DATAFRAME)

    # 6. Diff combinations vs completed runs
    completed_tuples = set()
    for run in completed_runs:
        combo_tuple = (run["dataset"], run["property"], run["representation"], run["kernel"], run["detrending"], run["owl"], run["elemental"], run["seed"])
        completed_tuples.add(combo_tuple)

    missing_commands = []
    for combo in all_combinations:
        combo_tuple = (combo["dataset"], combo["property"], combo["representation"], combo["kernel"], combo["detrending"], combo["owl"], combo["elemental"], combo["seed"])
        if combo_tuple not in completed_tuples:
            dataset_file = datasets[combo["dataset"]]["file"]
            file_hash = uuid.uuid4().hex
            sub_dir = file_hash[:2]
            temp_filename = f"{file_hash}.json"
            temp_filepath = os.path.join(ARCHIVE_DIR, sub_dir, temp_filename)
            os.makedirs(os.path.join(ARCHIVE_DIR, sub_dir), exist_ok=True)
            
            owl_flag = f"--owl {combo['owl']} " if combo['owl'] else ""
            detrend_flag = f"--detrending {combo['detrending']} " if combo['detrending'] else "--detrending '' "
            elemental_flag = "--elemental " if combo['elemental'] else "--no-elemental "
            if not combo['is_local']:
                elemental_flag = ""
            seed_flag = f"--seed {combo['seed']} "
            
            cmd = f"nc-krr {dataset_file} '{combo['property']}' {combo['representation']} {combo['kernel']} {detrend_flag}{elemental_flag}{owl_flag}{seed_flag}--archive {temp_filepath}"
            missing_commands.append(cmd)

    output_file = "missing_commands.txt"
    if missing_commands:
        print(f"# Missing {len(missing_commands)} combinations. Saving commands to {output_file}...")
        with open(output_file, "w") as f:
            for cmd in missing_commands:
                f.write(cmd + "\n")
        print(f"# Successfully generated {output_file}")
    else:
        print("# All combinations are complete!")
        if os.path.exists(output_file):
            with open(output_file, "w") as f:
                f.write("")

if __name__ == "__main__":
    main()
