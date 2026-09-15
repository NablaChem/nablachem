import streamlit as st
import plotly.graph_objects as go
import numpy as np

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
def render_heat_properties(experiment_data):
    if not experiment_data:
        return
        
    with st.container(border=True):
        st.write("### Property Performance Heatmap")
        st.write("Color represents `Null Model Error / Best Model Error`. Values > 1 indicate the model is better than the null model.")
        
        datasets = sorted(list(set(d.get("dataset") for d in experiment_data if d.get("dataset"))))
        reps = sorted(list(set(d.get("rep") for d in experiment_data if d.get("rep"))))
        kernels = sorted(list(set(d.get("kernel") for d in experiment_data if d.get("kernel"))))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sel_dataset = st.selectbox("Select Dataset", datasets, key="hp_dataset")
        with col2:
            sel_rep = st.selectbox("Select Representation", reps, key="hp_rep")
        with col3:
            sel_kernel = st.selectbox("Select Kernel", kernels, key="hp_kernel")
            
        # Filter data
        filtered = [d for d in experiment_data if d["dataset"] == sel_dataset and d["rep"] == sel_rep and d.get("kernel") == sel_kernel]
        
        if not filtered:
            st.info("No data available for this selection.")
            return
            
        # Extract properties and training points
        properties = []
        train_sizes = set()
        
        # Precompute matrix data
        matrix_data = {}
        
        for entry in filtered:
            prop = entry["prop"]
            properties.append(prop)
            
            # Find best learning curve values per training point
            best_rmses = {}
            null_rmse = None
            
            for run_lc in entry.get("runs", []):
                if not run_lc: continue
                
                # Get null model (ntrain == 1)
                nm_step = next((s for s in run_lc if s["ntrain"] == 1), None)
                if nm_step and null_rmse is None:
                    null_rmse = nm_step.get("test_rmse")
                
                # Get valid points
                for step in run_lc:
                    ts = step["ntrain"]
                    if ts > 1:
                        train_sizes.add(ts)
                        err = step.get("test_rmse")
                        if err is not None:
                            if ts not in best_rmses or err < best_rmses[ts]:
                                best_rmses[ts] = err
            
            if null_rmse is not None and best_rmses:
                for ts, best_err in best_rmses.items():
                    if best_err > 0:
                        ratio = null_rmse / best_err
                        if prop not in matrix_data: matrix_data[prop] = {}
                        matrix_data[prop][ts] = ratio

        properties = sorted(list(set(properties)))
        train_sizes = sorted(list(train_sizes))
        
        if not properties or not train_sizes:
            st.info("Insufficient learning curve data.")
            return
            
        # Build Z matrix (Y = properties, X = train_sizes)
        z_data = []
        text_data = []
        
        for prop in properties:
            row_z = []
            row_text = []
            for ts in train_sizes:
                val = matrix_data.get(prop, {}).get(ts, None)
                row_z.append(val)
                row_text.append(f"{val:.2f}" if val is not None else "N/A")
            z_data.append(row_z)
            text_data.append(row_text)
            
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[str(ts) for ts in train_sizes],
            y=properties,
            text=text_data,
            texttemplate="%{text}",
            colorscale="Viridis",
            hoverongaps=False,
            hovertemplate="Property: %{y}<br>Training Size: %{x}<br>Ratio: %{z:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            xaxis_title="Training Set Size",
            xaxis=dict(type='category'),
            yaxis_title="Property",
            height=max(300, len(properties) * 50 + 150),
            margin=dict(t=30, b=30, l=100, r=30)
        )
        
        st.plotly_chart(fig, width='stretch')
