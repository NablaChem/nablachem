import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys
import os
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')))
from nablachem.krr.vis import create_combined_learning_curve_plot

# Design constants
PALETTE = ["#1F3664", "#D59131", "#931621", "#2C8C99", "#42D9C8", "#8A2BE2", "#FF4500", "#2E8B57"]
DARK_GREY = "#202124"
FONT_FAMILY = "'Nunito', sans-serif"

def _apply_layout(fig):
    fig.update_layout(
        xaxis_type="log", yaxis_type="log",
        xaxis_title="Training Set Size", yaxis_title="Test RMSE",
        height=400, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, color=DARK_GREY)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0', zeroline=False)

@st.cache_data(show_spinner=False)
def get_cached_curve_fig(rep, prop, dataset, _runs):
    by_ntrain = defaultdict(lambda: defaultdict(list))
    for run_lc in _runs:
        if not run_lc: continue
        for entry in run_lc:
            n = entry["ntrain"]
            if n == 1: continue
            for metric in ("test_rmse", "val_rmse", "test_mae", "val_mae"):
                if metric in entry: by_ntrain[n][metric].append(entry[metric])
                
    stats = {}
    for n, metrics in by_ntrain.items():
        stats[n] = {}
        for metric, values in metrics.items():
            arr = np.array(values)
            stats[n][metric] = {
                "median": float(np.median(arr)),
                "p33": float(np.percentile(arr, 33)),
                "p67": float(np.percentile(arr, 67)),
            }
            
    if stats:
        curve_fig = create_combined_learning_curve_plot(stats)
        return curve_fig
    return None

# Fallback for dialog
if hasattr(st, "dialog"):
    dialog_decorator = st.dialog
elif hasattr(st, "experimental_dialog"):
    dialog_decorator = st.experimental_dialog
else:
    def dialog_decorator(title, width="large"):
        def decorator(func):
            def wrapper(*args, **kwargs):
                with st.container(border=True):
                    st.write(f"### {title}")
                    func(*args, **kwargs)
            return wrapper
        return decorator

@dialog_decorator("Learning Curves", width="large")
def render_learning_curves(experiment_data, lit_data):
    selected = st.session_state.get('selected_plots', [])
    if not selected:
        return
        
    for plot_config in selected:
        parts = [p.strip() for p in plot_config.split("|")]
        if len(parts) != 3: continue
        sel_rep, sel_prop, sel_dataset = parts
        
        matched_data = next((d for d in experiment_data if d["rep"] == sel_rep and d["prop"] == sel_prop and d["dataset"] == sel_dataset), None)
        if not matched_data or not matched_data.get("runs"): continue
        
        curve_fig = get_cached_curve_fig(sel_rep, sel_prop, sel_dataset, matched_data["runs"])
        
        if curve_fig:
            st.write(f"**{sel_rep} | {sel_prop} | {sel_dataset}**")
            st.plotly_chart(curve_fig, width="stretch", config={'displayModeBar': False}, key=f"modal_lc_{sel_rep}_{sel_prop}_{sel_dataset}")
        else:
            st.info("No valid runs found for the selected models.")

# Fallback for fragment
if hasattr(st, "fragment"):
    fragment_decorator = st.fragment
elif hasattr(st, "experimental_fragment"):
    fragment_decorator = st.experimental_fragment
else:
    def fragment_decorator():
        def decorator(func):
            return func
        return decorator

@fragment_decorator()
def render_all_learning_curves_bottom(experiment_data):
    st.write("---")
    st.header("Learning Curves", anchor="learning-curves")
    st.markdown("Mean learning curves for all models in the experimental matrix.")
    
    if not experiment_data:
        st.info("No experiment data available.")
        return
        
    properties = sorted(list(set(d.get("prop") for d in experiment_data if d.get("prop"))))
    reps = sorted(list(set(d.get("rep") for d in experiment_data if d.get("rep"))))
    datasets = sorted(list(set(d.get("dataset") for d in experiment_data if d.get("dataset"))))

    with st.form(key="lc_filter_form"):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 0.2])
        with col1:
            selected_props = st.multiselect("Properties", properties, default=properties, key="lc_props")
        with col2:
            selected_reps = st.multiselect("Representations", reps, default=reps, key="lc_reps")
        with col3:
            selected_datasets = st.multiselect("Datasets", datasets, default=datasets, key="lc_datasets")
        with col4:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            st.form_submit_button("", icon=":material/refresh:", type="tertiary", help="Refresh")
        
    cols = st.columns(2)
    valid_plots = 0
    
    for d in experiment_data:
        rep = d.get("rep", "Unknown")
        prop = d.get("prop", "Unknown")
        dataset = d.get("dataset", "Unknown")
        
        if prop not in selected_props or rep not in selected_reps or dataset not in selected_datasets:
            continue
            
        title = f"{rep} | {prop} | {dataset}"
        
        if not d.get("runs"): continue
        
        curve_fig = get_cached_curve_fig(rep, prop, dataset, d["runs"])
        
        if curve_fig:
            with cols[valid_plots % 2]:
                with st.container(border=True):
                    st.write(f"**{title}**")
                    # Tweak layout to fit well in grid (create a copy to not modify the cached fig layout permanently)
                    fig_copy = go.Figure(curve_fig)
                    fig_copy.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
                    # Streamlit >= 1.30 deprecates use_container_width and requires unique keys for identical figures
                    unique_key = f"lc_grid_{rep}_{prop}_{dataset}_{valid_plots}"
                    st.plotly_chart(fig_copy, width="stretch", config={'displayModeBar': False}, key=unique_key)
            valid_plots += 1