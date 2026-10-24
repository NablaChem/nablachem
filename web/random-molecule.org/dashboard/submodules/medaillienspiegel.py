import streamlit as st
import pandas as pd
from collections import defaultdict

def render_medaillienspiegel(experiment_data):
    if not experiment_data:
        return
        
    with st.container(border=True):
        st.write("### Medaillienspiegel")
        st.write("Displays how often each representation scored first, second, or third place in each property calculation (across all datasets).")
        
        kernels = sorted(list(set(d.get("kernel") for d in experiment_data if d.get("kernel"))))
        
        all_train_points = set()
        for d in experiment_data:
            for run in d.get("runs", []):
                for step in run:
                    if step.get("ntrain", 0) > 1:
                        all_train_points.add(step["ntrain"])
        train_points = sorted(list(all_train_points))
        
        errors = ["MAE", "RMSE"]
        
        if not kernels: kernels = ["Unknown"]
        if not train_points: train_points = ["All"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_kernel = st.selectbox("Kernel", kernels, key="ms_kernel")
        with col2:
            selected_train = st.selectbox("Training Points", train_points, key="ms_train")
        with col3:
            selected_error = st.selectbox("Error Metric", errors, key="ms_error")
            
        tasks = defaultdict(dict)
        err_key = "test_mae" if selected_error == "MAE" else "test_rmse"
        
        for d in experiment_data:
            ds = d.get("dataset")
            prop = d.get("prop")
            rep = d.get("rep")
            kernel = d.get("kernel")
            
            if not (ds and prop and rep): continue
            if kernel != selected_kernel: continue
            
            best_error = float('inf')
            
            if selected_train == "All":
                val = d.get(selected_error)
                if val is not None:
                    best_error = val
            else:
                for run in d.get("runs", []):
                    for step in run:
                        if step.get("ntrain") == selected_train:
                            val = step.get(err_key)
                            if val is not None and val < best_error:
                                best_error = val
                                
            if best_error != float('inf'):
                key = (ds, prop)
                if rep not in tasks[key] or best_error < tasks[key][rep]:
                    tasks[key][rep] = best_error
                    
        # Now, tally medals for each representation
        medals = defaultdict(lambda: {"🥇 Gold": 0, "🥈 Silver": 0, "🥉 Bronze": 0})
        all_reps = set()
        
        for key, rep_rmses in tasks.items():
            # Sort representations by RMSE (lowest is best)
            sorted_reps = sorted(rep_rmses.items(), key=lambda item: item[1])
            
            for rep, _ in sorted_reps:
                all_reps.add(rep)
                
            if len(sorted_reps) >= 1:
                medals[sorted_reps[0][0]]["🥇 Gold"] += 1
            if len(sorted_reps) >= 2:
                medals[sorted_reps[1][0]]["🥈 Silver"] += 1
            if len(sorted_reps) >= 3:
                medals[sorted_reps[2][0]]["🥉 Bronze"] += 1
                
        # Create a dataframe
        records = []
        for rep in all_reps:
            g = medals[rep]["🥇 Gold"]
            s = medals[rep]["🥈 Silver"]
            b = medals[rep]["🥉 Bronze"]
            total = g + s + b
            
            records.append({
                "Representation": rep,
                "🥇 Gold": g,
                "🥈 Silver": s,
                "🥉 Bronze": b,
                "Total Medals": total
            })
            
        if not records:
            st.info("No data available to calculate Medaillienspiegel.")
            return
            
        df = pd.DataFrame(records)
        
        # Sort by Gold, then Silver, then Bronze, then Total
        df = df.sort_values(by=["🥇 Gold", "🥈 Silver", "🥉 Bronze", "Total Medals"], ascending=False).reset_index(drop=True)
        
        st.dataframe(df, width='stretch')
