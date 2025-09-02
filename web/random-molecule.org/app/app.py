import streamlit as st
import nablachem.space as ncs
import leruli as lrl
import re
import base64

primary = "#1F3664"
secondary = "#dd9633"

st.set_page_config(
    page_title="Random molecules",
    layout="wide",
)
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 0}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def parse_formula(formula: str) -> dict[str, int]:
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    element_counts = {}
    for element, count in matches:
        count = int(count) if count else 1
        element_counts[element] = count
    return element_counts


def counts_to_criterion(counts: dict[str, int]) -> str:
    criterion = " & ".join(
        [f"{element} = {count}" for element, count in counts.items()]
    )
    return criterion, sum(counts.values())


def try_as_formula(molinput: str):
    # TODO: check if the elements in there are valid
    result = lrl.canonical_formula(molinput)
    if "formula" in result:
        element_counts = parse_formula(result["formula"])
        return counts_to_criterion(element_counts), set(element_counts.keys())


def parse_chemspace(space: str):
    chemspace = {}
    for line in space.split("\n"):
        element, valences = line.split(":")
        valences = [int(v) for v in valences.split(",")]
        chemspace[element] = valences
    return chemspace


def chemspace_to_string(chemspace: dict[str, list[int]]) -> str:
    spacecompact = " ".join(
        [
            f"{element}:{','.join([str(v) for v in valences])}"
            for element, valences in chemspace.items()
        ]
    )
    return spacecompact


st.write(
    "Molecules shown here are representative of the selected chemical space. Publication on arxiv: https://arxiv.org/abs/2508.20609. Contact: vonrudorff@uni-kassel.de."
)
tab_simple, tab_advanced = st.tabs([":bike: Simple", ":rocket: Advanced"])
criterion = None
chemspace = {
    "C": [4],
    "H": [1],
    "O": [2],
    "N": [3],
    "F": [1],
    "Cl": [1],
    "Br": [1],
    "I": [1],
    "P": [3, 5],
    "S": [2, 4],
    "Si": [4],
    "K": [1],
}


with tab_simple:
    st.write(
        "You can specify a chemical formula (e.g. `C8H10N4O2` or `(CH3)(CH2)8(CH3)`) a molecule name (`caffeine`) or a SMILES string (`CN1C=NC2=C1C(=O)N(C(=O)N2C)C`) and you will receive random molecules of the same chemical formula."
    )
    with st.form(key="input_form"):
        molinput = st.text_input(
            "Chemical Formula, Molecule name, or SMILES string.",
            key="sumformula",
            value="C8H10N4O2",
        )
        submit_button = st.form_submit_button("Generate")

    if submit_button:

        try:
            result = lrl.graph_to_formula(molinput)
            if "formula" in result:
                criterion = try_as_formula(result["formula"])
        except:
            pass
        if not criterion:
            criterion = try_as_formula(molinput)

        if not criterion:
            result = lrl.name_to_graph(molinput)
            if "graph" in result:
                molsmiles = result["graph"]
            else:
                molsmiles = molinput
            result = lrl.graph_to_formula(molsmiles)
            if "formula" in result:
                criterion, elements = try_as_formula(result["formula"])
        else:
            criterion, elements = criterion
        if not criterion:
            st.error("Did not recognize the input.")
        else:
            try:
                chemspace = {element: chemspace[element] for element in elements}
            except:
                criterion = None
                st.error(
                    "I don't know how many bonds some of the elements form. But you can tell me explicitly on the *Advanced* tab."
                )

with tab_advanced:
    with st.form(key="advanced_form"):
        space = st.text_area(
            "Search Space, one element per line, element followed by allowed valences.",
            value="C:4\nH:1\nO:2\nN:3\nF:1",
            height=200,
        )
        natoms = st.slider(
            "Select a number of atoms",
            min_value=2,
            max_value=30,
            value=29,
        )
        filterinput = st.text_input(
            "Filter condition", value="(C + O + N + F <= 9) & (C > 3)"
        )
        submit_button = st.form_submit_button("Generate")
    if submit_button:
        chemspace = parse_chemspace(space)
        criterion = filterinput, natoms

# explain the criterion
if criterion:
    criterion, natoms = criterion
    spacecompact = chemspace_to_string(chemspace)

    space = ncs.SearchSpace(spacecompact)
    c = ncs.ApproximateCounter()
    q_criterion = ncs.Q(criterion)

    with st.expander("Code to run yourself"):
        st.code(
            f"""# pip install --upgrade nablachem
import nablachem.space as ncs
counter = ncs.ApproximateCounter(show_progress=False)
space = ncs.SearchSpace("{spacecompact}")
criterion = ncs.Q("{criterion}")
natoms = {natoms}

total_molecule_count = counter.count(space, natoms, criterion)
mols = ncs.random_sample(
    counter, space, natoms=natoms, nmols=3, selection=criterion
)""",
            language="python",
        )

    if natoms > 30:
        st.write(
            "Sorry, that is more than 30 atoms. Please run the code above on your machine."
        )
    else:

        total_molecule_count = c.count(space, natoms, q_criterion)
        st.write(
            f"I think there are {total_molecule_count:,} molecules with {natoms} atoms in the search space that satisfy the filter condition."
        )
        if total_molecule_count == 0:
            st.write("Since there are no such molecules, none will be generated.")
        else:
            rows = 3
            columns = 3
            nmols = rows * columns

            if natoms > 25:
                st.write(
                    "Sorry, for larger molecules, we can only offer one at a time. If you run the code above on your machine, you can get as many as you like."
                )
                nmols = 1
                st.write(f"Here is one random molecule.")
            else:
                st.write(f"Here are {nmols} random molecules.")

            mols = ncs.random_sample(
                c, space, natoms=natoms, nmols=nmols, selection=q_criterion
            )

            table_columns = st.columns(columns)

            for i, mol in enumerate(mols):
                with table_columns[i % columns]:
                    molimage = lrl.graph_to_image(mol.SMILES, format="SVG", angle=0)[
                        "image"
                    ]
                    molimage = base64.b64decode(molimage).decode("utf-8")
                    st.image(molimage, caption=mol.SMILES, use_container_width=True)
