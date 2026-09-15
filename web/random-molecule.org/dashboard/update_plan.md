# Ui
1) combined learning curves (minimal)
Dropdown Dataset, Property

Graph content: all representations each just test MAE and only best kernel (use validation MAE)

2) List properties which are too difficult to learn:
Nullmodel error divide model error at 10k -> rank the percentage -> lowest 1st place

For UI read dataframe and use pygwalker
# Datamanagment
- trainingsdata, must be power of 2 steps, up to 20k
- use pre-calc pandas dataframe instead of json
- no clustering of propties of different datasets
- use column name - not property name -> property column map later via meta data in jsonl. 
for dataframe :
- sort values via validation mae, report test mae
- group by representation and ntrain

# Improvements
- add error units
- add units for properties
- use full hash in archieve, use hash to link archive json to dataframe
- use 1st level of hash block as folders for archive data
- use uid
- new representations in nc-krr, check if it automatically updates
- add --detrending for permutuation
- use --elemental , --no-elemental for local
- use owl [HOMO] for homo or gap property, use owl [LUMO] for lumo or gap property. Only for local

# streamlit 

https://github.com/Socvest/streamlit-on-Hover-tabs
https://github.com/victoryhb/streamlit-option-menu
https://github.com/okld/streamlit-elements
https://github.com/ObservedObserver/streamlit-shadcn-ui
https://echarts.streamlit.app/examples?ex_demo=Gradient+Stacked+Area+Chart
https://extras.streamlit.app/?extra=%F0%9F%83%8F+Card+Selector&ref=streamlit-io-component-all
https://pandas.pydata.org/docs/user_guide/index.html
https://pandas.pydata.org/docs/user_guide/index.html
https://docs.kanaries.net/gallery
https://ui.shadcn.com/

# nc-krr
(later) request feature via github: limit RAM usage. 

Usage: nc-krr [OPTIONS] JSONL_PATH COLUMN_NAME REPRESENTATION_NAME KERNEL_NAME

  Train KRR models on molecular data.

  JSONL_PATH: Path to gzipped JSONL file containing molecular data
  COLUMN_NAME: Property expression to predict using pandas DataFrame.eval()
  syntax.             Can be a simple column name like 'energy' or a
  calculated expression             like 'energy - baseline' or 'E_high -
  E_low'. For column names with             special characters (dashes,
  spaces), use backticks like `E-high` - `E-low`. REPRESENTATION_NAME: Name of
  the molecular representation to use.                  Built-in
  representations: FCHL19Global, FCHL19Local, MACEMPGlobal, MACEMPLocal,
  MACEOFFGlobal, MACEOFFLocal, MACEOMolGlobal, MACEOMolLocal, MBDFGlobal,
  MBDFLocal, SLATMGlobal, SLATMLocal, cMBDFGlobal, cMBDFLocal
  Custom representations can be loaded from any importable module using
  dotted notation, e.g. 'mymodule.MyRepresenter' or 'pkg.sub.MyRep'.
  The class must implement the BaseRepresenter interface (compute/build).
  KERNEL_NAME: Name of the kernel function to use.          Available kernels:
  Bump, Exponential, Gaussian, GeneralizedCauchy, InverseMultiquadric,
  InverseQuadratic, Matern32, Matern52, MaternGeneral, Polynomial, Power,
  RationalQuadratic, Sigmoid, WendlandK0, WendlandK1, WendlandK2, WendlandK3,
  WendlandK4, WuC2, WuC4, WuC6

  The dataset is split with the first maxcount molecules used for training,
  and the remaining molecules used as holdout/test data.

Options:
  --limit INTEGER               Maximum number of molecules to load (includes
                                training + holdout). Defaults to maxcount +
                                2000
  --mincount INTEGER            Minimum training size
  --maxcount INTEGER            Maximum training size (rest used as holdout)
  --select TEXT                 Selection expression for filtering dataset
                                rows
  --detrending TERMS            Comma-separated detrending terms, e.g.
                                'atomic,charge'. Available: atomic, pairs,
                                charge, spin. Pass an empty string to disable
                                detrending. [default: atomic]
  --holdout-residuals TEXT      Output JSONL file path for holdout residuals
  --elemental / --no-elemental  Mask cross-element atom pairs in local kernel
                                (requires local representation)
  --alchemical PATH             JSON file with per-element-pair weights
                                {"Z1,Z2": float} (Z1<=Z2). Requires local
                                representation.
  --owl HOMO            Orbital-weighted learning: weight every atom
                                of a local representation by its Mulliken
                                population in the given GFN2-xTB orbital, then
                                sum into a global representation.
  --archive TEXT                Output file for KRR archive data
  --seed INTEGER                Random seed for numpy. Use -1 (default) for
                                non-deterministic runs, or a non-negative
                                integer for reproducible shuffles.
  --predict PATH                JSONL file with an 'xyz' column to predict in
                                place. Its molecules are appended as an
                                extended holdout and predicted with the
                                largest trained model; a 'property' column is
                                written back into the file.

