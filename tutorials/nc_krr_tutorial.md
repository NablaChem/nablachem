# Shell, Terminal, and CLI

* **Shell:** A command-line interpreter (the software that processes your commands).
* **Terminal:** The emulator/window that allows you to interact with the shell.
* **CLI:** The Command-Line Interface (the overall method of interaction).
* **IDE:** Integrated Development Environment (e.g., VS Code, PyCharm, Google Colab). These provide a graphical user interface to manage projects, edit code, and handle environments.

### Platform Differences

* **Windows:** PowerShell (modern) or Command Prompt (`cmd`).
* **Linux/macOS:** Terminal (e.g., GNOME Terminal, iTerm2).

---
# Essential Terminal Tips

## Essential Terminal Commands

| Action | Linux / macOS | Windows (CMD) | Description & Example |
| :--- | :--- | :--- | :--- |
| **Print Directory** | `pwd` | `cd` | Shows the exact folder path you are currently in. |
| **List Files** | `ls` | `dir` | Lists all files and folders in your current directory. |
| **Change Directory** | `cd` | `cd` | Moves you into a different folder. <br>*Example:* `cd Desktop` |
| **Go Back** | `cd ..` | `cd ..` | The two dots `..` mean "the folder one level up". This moves you out of the current folder. |
| **Make Directory** | `mkdir` | `mkdir` | Creates a new, empty folder. <br>*Example:* `mkdir my_project` |
| **Move / Rename** | `mv` | `move` | Moves a file to a new location, OR renames it if you keep it in the same folder. <br>*Example:* `mv old.txt new.txt` |
| **Copy** | `cp` | `copy` | Creates a duplicate of a file. <br>*Example:* `cp data.csv backup.csv` |
| **Delete File** | `rm` | `del` | Permanently deletes a file. **Warning:** This does not go to the Recycle Bin! <br>*Example:* `rm unwanted.txt` |
| **Clear Screen** | `clear` | `cls` | Wipes the terminal screen clean so you can start fresh without clutter. |
|

* **Cancel Command:** `Ctrl + C`
* **Execute:** `Enter`
* **Command History:** `Up/Down` arrow keys
* **VS Code Shortcut:** Navigate to folder, then type `code .`

## More specfic commands

| Tool | Command | Description | Example |
| :--- | :--- | :--- | :--- |
| **Conda** | `conda create --name <env> python=<ver>` | Creates a new, isolated Python environment. | `conda create --name my_env python=3.10` |
| **Conda** | `conda activate <env>` | Turns on a specific environment. Do this before running code. | `conda activate my_env` |
| **Conda** | `conda deactivate` | Exits the current environment and returns to base. | `conda deactivate` |
| **Conda** | `conda env list` | Lists all environments on your computer. | `conda env list` |
| **Pip** | `pip install <package>` | Installs an external library into your active environment. | `pip install nablachem` |
| **Pip** | `pip list` | Shows every library currently installed in your environment. | `pip list` |
| **Python** | `python <script.py>` | Executes a Python file. | `python my_script.py` |
| **Python** | `python -m <module>` | Locates an installed library module and runs it as a script. | `python -m pytest` |
| **nc-krr** | `nc-krr <args>` | Custom executable for Kernel Ridge Regression. Requires pip installation. | `nc-krr data.jsonl energy FCHL19Global Gaussian` |
| **Git** | `git clone <url>` | Downloads a repository from GitHub directly to your computer. | `git clone https://github.com/...` |
| **Git** | `git status` | Shows which files you have modified or added. | `git status` |
| **Git** | `git add <file>` | Stages a file so it will be included in your next save (commit). | `git add my_script.py` |
| **Git** | `git commit -m "<msg>"` | Saves your staged changes locally with a descriptive message. | `git commit -m "Fix data loader"` |
| **Git** | `git push` | Uploads your saved commits to the remote GitHub repository. | `git push` |
| **Git** | `git pull` | Downloads the latest changes from GitHub to your computer. | `git pull` |
|

* **Password Prompt:** Characters are hidden for security (do not expect to see `*` or characters move).
* **Flags** you can use flags to specify inputs for your function when using the CLI. There exists short flags `-` and long flags `--`. Short flags are usually just an abbreviation of the long flag. 

## Standard Command-Line Flags

| Short | Long | Target | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `-m` | | Python | Locates a module in your installed environment and runs it as a script. | `python -m nablachem.krr.cli` |
| `-c` | | Python | Executes a quick string of Python code directly in the terminal without a file. | `python -c "print('Ready')"` |
| `-i` | | Python | Drops you into an interactive Python shell immediately after a script finishes running. | `python -i run_krr.py` |
| `-V` | `--version` | Python | Prints the exact version of Python currently active in your environment. | `python -V` |
| `-v` | `--verbose` | Script | A common standard flag to increase the detail of log messages printed to the screen. | `pytest -v` |
| `-h` | `--help` | Script | Prints the manual, listing all required arguments and optional flags for a script. | `python run_krr.py --help` |
|

# Data Formats

### JSON (JavaScript Object Notation)

A text-based format for data exchange, structured like a Python dictionary.

```json
{
    "Name": "CO2",
    "coordinates": [[0.0, 0.0, 0.0], [1.163, 0.0, 0.0], [-1.163, 0.0, 0.0]],
    "element_order": [6, 8, 8],
    "tot_num_atoms": 3
}

```

### JSONL (JSON Lines)

A format where each line is an independent, valid JSON object. This is ideal for large datasets because you can process line-by-line without loading the entire file into RAM.

```json
{"Name": "CO2", "coordinates": [[0.0, 0.0, 0.0], [1.163, 0.0, 0.0], [-1.163, 0.0, 0.0]], "element_order": [6, 8, 8], "tot_num_atoms": 3}
{"Name": "H2O", "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]], "element_order": [1, 1, 8], "tot_num_atoms": 3}

```

---

# Environments & Packages

Environments isolate project dependencies to prevent version conflicts.

### Using Conda

* **Create:** `conda create --name my_env python=3.12`
+ **Activate:** `conda activate my_env`
+ **Deactivate:** `conda deactivate`

### Using venv (Built-in)

* **Create:** `python -m venv myenv`
+ **Activate:**
    + Windows: `myenv\Scripts\activate`
    + Linux/macOS: `source myenv/bin/activate`
+ **Deactivate:** `deactivate`

---

# WSL (Windows Subsystem for Linux)

WSL runs a Linux kernel inside Windows.

* **Access:** Type `wsl` in any Windows terminal to enter the Linux environment.
* **IDE Integration:** In VS Code, click the `><` icon in the bottom-left corner and select **Connect to WSL**.
* **Performance Tip:** Always store your code inside the Linux file system (e.g., `~/projects/`), *not* on the Windows `C:` drive.

---

# Nablachem KRR Workflow

There are two ways to use this tool: by writing a Python script (best for custom research) or by using the built-in CLI commands (best for quick training and standardized reporting).
You can receive help for this module by tipping this in yor terminal: `nc-krr --help`

## IDE Approach

1. **Project Setup:** 
    1. Open your project folder in your IDE (e.g., VS Code)
    2. Connect to WSL if needed.
    3. Create a new environment and activate it.
    4. ```pip install --upgrade nablachem```
2. **Script:** Create a new Python file (e.g., `my_script.py`). 
    1. Import needed submodules from _nablachem_
    2. Download a Dataset as ```.jsonl.gz```or ```.json```
    3. Find the available labels within the dataset 
        - They can be found in the dataset documentation
        - Or run the script with a guessed label, it will show the available labels then.

Example
```python
from nablachem.krr import dataset, features
# Load your training data
train_data = dataset.DataSet("qm9.jsonl.gz", labelname="HOMO_Energy", limit=1000)
```


3. **Define Representation:** Choose a "representer" to convert molecules into machine-readable format. Use `Global` versions for predicting molecular energy.
```python
representation = features.FCHL19Global()
```


4. **Build the Blueprint:** This scans your data to ensure the machine learning model gets uniform inputs.
```python
representation.build(train_data, compatible_to=[test_data])
```


5. **Compute Features:** Generate the mathematical vectors ($X$ arrays) for your model.
```python
x_train = representation.compute(train_data.molecules)
x_test = representation.compute(test_data.molecules)
```



## CLI Approach
0. Create you environment and install nablachem via pip. 
1. **Train the Model:** Open your terminal, ensure your environment is activated, and run the main script. This will perform the training and save the results to a file.
    ```bash
    nc-krr <jsonl_path> <column_name> <representation_name> <kernel_name> [options]
    ```
    ### Command-Line Arguments

    | Name | Description |
    | :--- | :--- |
    | `jsonl_path` | Path to gzipped JSONL file containing molecular data
    | `column_name` | Name of the label column in the JSONL file (what you want to predict)
    | `representation_name` | Name of the molecular representation to use, see table below or use nc-krr --help
    | `kernel_name` |
    |
    [options] are any flags specified in the table below (nc-krr Command-Line Options (Flags)).

    **Example**:
    ```bash
    nc-krr TM_GSspinPlus.jsonl.gz HOMO_Energy FCHL19Global Gaussian --archive my_results.json --no-detrend-atomic
    ```
    `--no-detrend-atomic` is part of [options]
    The order in which you write the flags does not matter here.
    | Option Flag | Default Value | Description |
    | :--- | :--- | :--- |
    | `--limit` | `maxcount + 2000` | Maximum total molecules to load (training + holdout). |
    | `--mincount` | `128` | Minimum number of molecules for the training learning curve. |
    | `--maxcount` | `2048` | Maximum number of training molecules (remainder used as holdout). |
    | `--select` | `None` | Pandas query expression for filtering dataset rows. |
    | `--detrend-atomic` / `--no-detrend-atomic` | `True` | Toggles atomic count linear detrending `True`means _enabled_, `False` means _disabled_. |
    | `--detrend-pairs` | `None` | Applies pairwise detrending (e.g., `gCP`). `True`means _enabled_, `False` means _disabled_. |
    | `--elemental` / `--no-elemental` | `False` | Masks cross-element pairs in local kernels (Local reps only). |
    | `--holdout-residuals` | `None` | File path to export holdout residuals as a JSONL. |
    | `--archive` | `archive.json` | Output JSON file for hyperparameter and learning curve data. |

    **Tip** Type `nc-krr --help`to get a list with the available `kernel_name`s and `representation_name`s.

    **Tip** Find the label names by typing something valid e.g. label in the place of the \<column_name\> to see a list hin the error message. You can find the availabel labels here:
    
    `[info     ] Dataset columns                columns=['refcode', 'total_charge', ...]`

    **Tip** You can exit the stream by disrupting the terminal with `ctrl c`, or just open a new terminal, activate the environment and work with that terminal, so that you can still see your stream.

    **What this does:** It tells the computer to use the `cli` module to read your data `TM_GSspinPlus.jsonl.gz`, calculate the `FCHL19` global representation, train a model using the `Gaussian` kernel, and save the performance statistics to `my_results.json`.
    

2. **Visualize Results:** Use the `vis.py` script to launch a web browser dashboard showing your learning curves and hyperparameter heatmaps.
    ```bash
    nc-krr-vis my_results.json
    ```
    **Tip**: accept the terms of streamlit in your terminal by hitting enter 2x.

    Type `nc-krr-vis --help`to get a list with explanaitions in your terminal.

    **What this does:** It opens a new tab in your web browser. Here, you can interactively click through tabs to see how well the model learned and where the optimal settings are.

3. **Visualize mean of multiple models**
    To get the mean of different models (same representation, same options) add to the `nc-krr-vis`executable all filepaths to the json files you want to compare. To get these json files look at step 1.
    ```bash
    nc-krr-vis my_results_0.json my_results_1.json my_results_2.json ...
    ```

### Label calculations 
The `nc-krr`module let's you modify the column labels (`column_name`), you want to use for prediction.
The _column_name_ can be a simple label like _energy_ or a calculated expression like 'energy - baseline' or 'E_high -E_low'. For column names with special characters (dashes,spaces), use backticks like \`E-high\` - \`E-low\`.

Examples:
| Correct | Incorrect |
| :--- | :--- |
HOMO_Energy | \`HOMO_Energy\` |
'HOMO_Energy'| |
HOMO_Energy*2 | |
'\`HOMO_Energy\` - \`LUMO_Energy\`' | HOMO_Energy - LUMO_Energy |
'HOMO_Energy - LUMO_Energy'| |
|

### Filter molecules with `--select`
You can filter for certain molecules by using the `--select` flag.
Example:
| Filter | Description |
| :--- | :--- |
| `--select 'n_atoms <= 3'`| Only include molecules consiting of less or equal to 3 atoms. |
| `--select 'n_C == 5'`| Select molecules with exactly 5 carbon atoms. |
| `--select 'n_Fe == 1 and n_C >= 5'`| Select molecules with exact 1 iron atom and at least 5 carbon atoms. |
| `--select 'HOMO_Energy < -5.0'`| Select all molecules with less than -5.0 units of HOMO energy.|
| `--select '(n_C / n_atoms) > 0.5'` | Select a carbon ratio of more than 50%. |
| `--select 'spin_multiplicity in [2, 4]'` | Selects only molecules with a spin multiplicity of 2 or 4. |
|

You can use for you select flag , _column_name_, _n_atoms_ and _n_X_ (_X_: any element) as a variable. You can use the following operators: comparisons (==, !=, <,<=, >,>=), booleans (and, or, not), math ( +, - , *, /, **) and memberships (in, not in). 

### Limit, mincount, maxcount
! If you select a --limit < 2048 without specifying --maxcount < --limit and mincount <= --maxcount, it will generate an error.
Example Error:
```bash
nc-krr TM_GSspinPlus.jsonl.gz HOMO_Energy FCHL19Global Gaussian --archive my_results.json --limit 100 --no-detrend-atomic
```
Example Fix: 
```bash
nc-krr TM_GSspinPlus.jsonl.gz HOMO_Energy FCHL19Global Gaussian --archive my_results.json --limit 100  --mincount 16 --maxcount 80 --no-detrend-atomic
```
! These example numbers are NOT recommended for actual calculations. Use these only test if the environment is corret and the nablachem installation is healthy.

The significance of --limit using the example above:
We have 10 molecules in genereal (--limit 100). From those we select an amount of atoms we want to train on to find the hyperparamters which is --maxcount. Usually you choose 80% for training and 20% for holdout. Therefore let's choose --maxcount 80 as our trainingsset.
From these 80 molecules we also do a 20%:80% split (this is hard-coded, you do not need to worry about it).Now we select 16 molecules (80*0.2) for testing and 64 for training for the hyperparamters. We do this process multiple times (~50 times) with the same 80 molecule we chose previously but we shuffle them around, so that a molecule which was previously in the test set might end up in the trainings set and vice versa.
After the hyperparamter was found we apply it to the 20 molecules in the holdout set.
To create a learning curve we need different amounts of molecules used for trainig. That is what the --mincount is for. If we start with --mincount 16 we use 16 molecules for testing and 84 as holdout. The next datapoint of the learning curve is then 32 molecules for testing and 68 as holdout.
The size of the trainingsset duplicates for each iteration starting from --mincount until --maxcount is reached as its final datapoint.
### Archive
The Archvive flag `--archive` + `_my_file_name_.json` generates a JSON file (with a name of your choice) which documents the learning steps for you. 
Here is a sketch of the generated  `_my_file_name_.json`:
```JSON
{
    "hyperopt" : 
    [
        {
        "ntrain": mincount,
        "sigma": float,
        "lambda": float,
        "val_rmse": [ float, ... , float]],
        "val_mae": [ float, ... , float],
        "train_rmse": [ float, ... , float],
        "train_mae": [ float, ... , float],
        }
        ..., #varies hyperparamters for each ntrain values - multiple dictionaries per same ntrain exits
        {
        "ntrain": maxcount,
        "sigma": float,
        "lambda": float,
        "val_rmse": [ float, ... , float],
        "val_mae": [ float, ... , float],
        "train_rmse": [ float, ... , float],
        "train_mae": [ float, ... , float],
        }
    ]
    "spectrum" : 
    {
    str(mincount) : [ float, ... , float],
    ...,
    str(maxcount) : [ float, ... , float],
    },
    "learning_curve" : 
    [
        {
        "ntrain": mincount,
        "val_rmse": float,
        "test_rmse": float,
        "val_mae": float,
        "test_mae": float,
        "hyperparameters": 
            {
            "sigma": float,
            "lambda": float
            }
        },
        ...,
        {
        "ntrain": maxcount,
        "val_rmse": float,
        "test_rmse": float,
        "val_mae": float,
        "test_mae": float,
        "hyperparameters": 
            {
            "sigma": float,
            "lambda": float
            }
        }
    ],
    "metadata" : 
    {
        "representation": str,
        "kernel": str,
        "detrend_atomic": bool,
        "detrend_pairs": null,
        "elemental": bool,
        "file_hash": str,
        "file_path": str,
        "column_name": str,
        "limit": int,
        "select": null
    }
}
```

---




