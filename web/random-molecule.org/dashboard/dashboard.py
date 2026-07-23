import streamlit as st
import os
import glob
import json

# Import graphic components
from pie_matrix import render_pie_matrix
from learning_curves import render_learning_curves
from leaderboard import render_leaderboard
from heatmap import render_heatmap
from heat_properties import render_heat_properties
from medaillienspiegel import render_medaillienspiegel
from lit_values_ui import render_lit_values_interface

# --- Helper Functions ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def parse_archives():
    results_db_path = "results_db.json"
    archive_files = []
    if os.path.exists(results_db_path):
        try:
            with open(results_db_path, "r") as f:
                db = json.load(f)
                archive_files = [x["file"] for x in db]
        except:
            archive_files = glob.glob("Archieve/*/*.json")
    else:
        archive_files = glob.glob("Archieve/*/*.json")
            
    grouped = {}
    all_reps = set()
    all_props = set()
    all_datasets = set()
    
    for rf in archive_files:
        try:
            with open(rf, 'r') as f:
                data = json.load(f)
                meta = data.get("metadata", {})
                lc = data.get("learning_curve", [])
                
                rep = meta.get("representation", "Unknown")
                prop = meta.get("column_name", "Unknown")
                
                # dataset is directory name
                dir_name = os.path.basename(os.path.dirname(rf))
                dataset = meta.get("dataset", dir_name)
                
                kernel = meta.get("kernel", "Gaussian")
                
                all_reps.add(rep)
                all_props.add(prop)
                all_datasets.add(dataset)
                
                combo = (rep, prop, dataset, kernel)
                if combo not in grouped:
                    grouped[combo] = []
                    
                grouped[combo].append(lc)
        except Exception as e:
            continue
            
    experiment_data = []
    completed_combinations = set()
    
    for (rep, prop, dataset, kernel), runs in grouped.items():
        completed_combinations.add((rep, prop, dataset))
        
        best_mae = float('inf')
        best_rmse = float('inf')
        max_train_size = 0
        
        for run_lc in runs:
            if not run_lc: continue
            valid_steps = [step for step in run_lc if step["ntrain"] > 1]
            if not valid_steps: continue
            
            last_step = max(valid_steps, key=lambda x: x["ntrain"])
            ts = last_step["ntrain"]
            if ts > max_train_size: max_train_size = ts
            
            if last_step.get("test_mae", float('inf')) < best_mae:
                best_mae = last_step["test_mae"]
            if last_step.get("test_rmse", float('inf')) < best_rmse:
                best_rmse = last_step["test_rmse"]
                
        experiment_data.append({
            "rep": rep,
            "prop": prop,
            "dataset": dataset,
            "kernel": kernel,
            "train_size": max_train_size,
            "MAE": best_mae if best_mae != float('inf') else None,
            "RMSE": best_rmse if best_rmse != float('inf') else None,
            "runs": runs
        })
        
    return sorted(list(all_reps)), sorted(list(all_props)), sorted(list(all_datasets)), completed_combinations, experiment_data

# --- Main App Initialization ---
st.set_page_config(layout="wide")
load_css("style.css")
st.title("Machine Learning Molecule Dashboard")

# --- Quick Navigation ---
nav_html = """
<div style="display: flex; gap: 10px; margin-bottom: 20px;">
    <a href="#leaderboard" style="padding: 8px 16px; background-color: #f0f2f6; color: #31333F; border-radius: 5px; text-decoration: none; font-weight: 600; border: 1px solid #dcdcdc; transition: background-color 0.2s;">Leaderboard</a>
    <a href="#medaillienspiegel" style="padding: 8px 16px; background-color: #f0f2f6; color: #31333F; border-radius: 5px; text-decoration: none; font-weight: 600; border: 1px solid #dcdcdc; transition: background-color 0.2s;">Medaillienspiegel</a>
    <a href="#experimental-matrix" style="padding: 8px 16px; background-color: #f0f2f6; color: #31333F; border-radius: 5px; text-decoration: none; font-weight: 600; border: 1px solid #dcdcdc; transition: background-color 0.2s;">Matrix</a>
    <a href="#learning-curves" style="padding: 8px 16px; background-color: #f0f2f6; color: #31333F; border-radius: 5px; text-decoration: none; font-weight: 600; border: 1px solid #dcdcdc; transition: background-color 0.2s;">Learning Curves</a>
    <a href="#performance-heatmap" style="padding: 8px 16px; background-color: #f0f2f6; color: #31333F; border-radius: 5px; text-decoration: none; font-weight: 600; border: 1px solid #dcdcdc; transition: background-color 0.2s;">Heatmap</a>
    <a href="#property-performance-heatmap" style="padding: 8px 16px; background-color: #f0f2f6; color: #31333F; border-radius: 5px; text-decoration: none; font-weight: 600; border: 1px solid #dcdcdc; transition: background-color 0.2s;">Property Heatmap</a>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# --- Session State Memory ---
if 'selected_plots' not in st.session_state:
    st.session_state.selected_plots = []
if 'dimmed_reps' not in st.session_state:
    st.session_state.dimmed_reps = []
if 'dimmed_props' not in st.session_state:
    st.session_state.dimmed_props = []

# --- Render Sidebars ---
lit_data = render_lit_values_interface()

# --- Data Loading ---
all_reps, all_props, datasets, completed_combinations, experiment_data = parse_archives()

if not all_reps:
    st.warning("No data found in Archieve directory. Please run `dashboard_manager.py` and then execute `nc-krr` commands to generate results.")
else:
    # --- Execute Graphic Components ---
    render_leaderboard(experiment_data)
    render_medaillienspiegel(experiment_data)

    render_pie_matrix(
        all_reps=all_reps, 
        all_props=all_props, 
        datasets=datasets, 
        completed_combinations=completed_combinations, 
        json_directory="Archieve"
    )

    # Compile current native Plotly selections into selected_plots
    active_selections = []
    for rep in all_reps:
        for prop in all_props:
            key = f"{rep}_{prop}"
            if key in st.session_state:
                selection = st.session_state[key].get("selection", {})
                for pt in selection.get("points", []):
                    dataset = pt.get("label")
                    if dataset:
                        active_selections.append(f"{rep} | {prop} | {dataset}")
    st.session_state.selected_plots = active_selections

    render_learning_curves(experiment_data, lit_data)

    render_heatmap(experiment_data)
    render_heat_properties(experiment_data)