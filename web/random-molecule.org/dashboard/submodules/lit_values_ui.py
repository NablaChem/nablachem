import streamlit as st
import json
import os

LIT_VALUES_FILE = "lit_values.json"

def load_lit_values():
    if os.path.exists(LIT_VALUES_FILE):
        with open(LIT_VALUES_FILE, 'r') as f:
            return json.load(f)
    return {"datasets": {}}

def save_lit_values(data):
    with open(LIT_VALUES_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def render_lit_values_interface():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Literature Values")
    st.sidebar.markdown("Provide literature values (baselines) for comparison in the learning curves.")
    
    lit_data = load_lit_values()
    
    with st.sidebar.expander("Manage Literature Values", expanded=False):
        uploaded_file = st.file_uploader("Upload Lit Values JSON", type=['json'])
        if uploaded_file is not None:
            try:
                new_data = json.load(uploaded_file)
                if "datasets" in new_data:
                    # Merge datasets
                    for ds, props in new_data["datasets"].items():
                        if ds not in lit_data["datasets"]:
                            lit_data["datasets"][ds] = {}
                        for prop, metrics in props.items():
                            lit_data["datasets"][ds][prop] = metrics
                    save_lit_values(lit_data)
                    st.success("Successfully imported literature values!")
                else:
                    st.error("Invalid JSON structure. Expected a 'datasets' key.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
        st.markdown("**Current Data:**")
        st.json(lit_data)
        
        # Simple manual entry form
        st.markdown("**Add Manual Entry**")
        with st.form("manual_lit_entry"):
            ds_name = st.text_input("Dataset Name")
            prop_name = st.text_input("Property")
            mae_val = st.number_input("MAE (optional)", value=0.0, format="%.4f")
            rmse_val = st.number_input("RMSE (optional)", value=0.0, format="%.4f")
            source = st.text_input("Source/Paper")
            
            submitted = st.form_submit_button("Add Value")
            if submitted:
                if ds_name and prop_name:
                    if ds_name not in lit_data["datasets"]:
                        lit_data["datasets"][ds_name] = {}
                    
                    entry = {}
                    if mae_val > 0: entry["MAE"] = mae_val
                    if rmse_val > 0: entry["RMSE"] = rmse_val
                    if source: entry["source"] = source
                    
                    lit_data["datasets"][ds_name][prop_name] = entry
                    save_lit_values(lit_data)
                    st.success("Added new literature value!")
                    st.rerun()
                else:
                    st.error("Dataset Name and Property are required.")
                    
    return lit_data
