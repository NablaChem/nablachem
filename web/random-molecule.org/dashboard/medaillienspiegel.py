import streamlit as st
import pandas as pd
from collections import defaultdict

def render_medaillienspiegel(experiment_data):
    if not experiment_data:
        return
        
    with st.container(border=True):
        st.write("### Medaillienspiegel")
        st.write("Displays how often each representation scored first, second, or third place in each property calculation (across all datasets).")
        
        # We need to rank representations for each (dataset, property) combination based on best RMSE
        # First, group data by (dataset, property)
        # Inside each group, keep track of the best RMSE per representation
        
        tasks = defaultdict(dict)
        
        for d in experiment_data:
            ds = d.get("dataset")
            prop = d.get("prop")
            rep = d.get("rep")
            rmse = d.get("RMSE")
            
            if ds and prop and rep and rmse is not None:
                # If there are multiple kernels/runs, we take the best RMSE for that representation
                key = (ds, prop)
                if rep not in tasks[key] or rmse < tasks[key][rep]:
                    tasks[key][rep] = rmse
                    
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
        
        st.dataframe(df, use_container_width=True)
