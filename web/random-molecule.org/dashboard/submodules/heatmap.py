import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Design constants
FONT_FAMILY = "'Nunito', sans-serif"
DARK_GREY = "#202124"

# Standard palette for real data points
REAL_COLORSCALE = [
    [0.0, "#1F3664"],
    [1.0, "#42D9C8"]
]

# Greyscale palette for interpolated/estimated data points
INTERPOLATED_COLORSCALE = [
    [0.0, "#343434"],
    [1.0, "#AAAAAA"]
]

def _get_metric_at_ts(entry, ts, metric_name):
    if not entry: return None
    key = "test_mae" if metric_name == "MAE" else "test_rmse"
    best_val = float('inf')
    for run in entry.get("runs", []):
        for step in run:
            if step.get("ntrain") == ts:
                val = step.get(key)
                if val is not None and val < best_val:
                    best_val = val
    return best_val if best_val != float('inf') else None

def _interpolate(experiment_data, ds, rep, prop, selected_kernel, slider_val, available_train_sizes, min_train, max_train, metric):
    entry = next((d for d in experiment_data if d.get("rep") == rep and d.get("dataset") == ds and d.get("prop") == prop and d.get("kernel") == selected_kernel), None)
    if not entry: return None

    if slider_val in available_train_sizes:
        val = _get_metric_at_ts(entry, slider_val, metric)
        if val is not None: return val
        
    lower_ts = max([ts for ts in available_train_sizes if ts <= slider_val], default=min_train)
    upper_ts = min([ts for ts in available_train_sizes if ts >= slider_val], default=max_train)
    
    val_lower = _get_metric_at_ts(entry, lower_ts, metric)
    val_upper = _get_metric_at_ts(entry, upper_ts, metric)
    
    if val_lower is not None and val_upper is not None:
        if lower_ts == upper_ts or val_lower <= 0 or val_upper <= 0:
            return val_lower
        else:
            log_lower_x, log_upper_x, log_slider_x = np.log(lower_ts), np.log(upper_ts), np.log(slider_val)
            log_lower_y, log_upper_y = np.log(val_lower), np.log(val_upper)
            factor = (log_slider_x - log_lower_x) / (log_upper_x - log_lower_x)
            return np.exp(log_lower_y + factor * (log_upper_y - log_lower_y))
    return val_lower if val_lower is not None else val_upper

@st.cache_data(show_spinner=False)
def precalculate_frames(experiment_data, prop, selected_kernel, reps, datasets, min_train, max_train, available_train_sizes):
    frames_data = {}
    
    # Use only the actual available training points
    slider_steps = list(available_train_sizes)
    
    for val in slider_steps:
        is_real = val in available_train_sizes
        
        z_mae, text_mae = [], []
        z_rmse, text_rmse = [], []
        
        for rep in reps:
            z_row_mae, text_row_mae = [], []
            z_row_rmse, text_row_rmse = [], []
            
            for ds in datasets:
                val_mae = _interpolate(experiment_data, ds, rep, prop, selected_kernel, val, available_train_sizes, min_train, max_train, "MAE")
                z_row_mae.append(val_mae)
                text_row_mae.append(f"{val_mae:.4f}" if val_mae is not None else "N/A")
                
                val_rmse = _interpolate(experiment_data, ds, rep, prop, selected_kernel, val, available_train_sizes, min_train, max_train, "RMSE")
                z_row_rmse.append(val_rmse)
                text_row_rmse.append(f"{val_rmse:.4f}" if val_rmse is not None else "N/A")
                
            z_mae.append(z_row_mae)
            text_mae.append(text_row_mae)
            z_rmse.append(z_row_rmse)
            text_rmse.append(text_row_rmse)
            
        frames_data[val] = {
            'is_real': is_real,
            'MAE': {'z': z_mae, 'text': text_mae},
            'RMSE': {'z': z_rmse, 'text': text_rmse}
        }
        
    return slider_steps, frames_data

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
def render_heatmap(experiment_data):
    with st.container(border=True):
        st.write("### Error Heatmaps")
        
        datasets = sorted(list(set(d.get("dataset") for d in experiment_data if d.get("dataset"))))
        reps = sorted(list(set(d.get("rep") for d in experiment_data if d.get("rep"))))
        properties = sorted(list(set(d.get("prop") for d in experiment_data if d.get("prop"))))
        kernels = sorted(list(set(d.get("kernel") for d in experiment_data if d.get("kernel"))))

        if not properties: properties = ["Unknown"]
        if not kernels: kernels = ["Unknown"]

        # Control Menu
        col1, col2 = st.columns(2)
        with col1:
            selected_props = st.multiselect("Properties", properties, default=properties, key="hm_props")
        with col2:
            selected_kernel = st.selectbox("Kernel", kernels, key="hm_kernel")

        if not selected_props:
            st.info("Please select at least one property to display heatmaps.")
            return

        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

        for prop in selected_props:
            st.markdown(f"#### {prop}")
            
            # Extract available unique training points for this specific property
            train_sizes_set = set()
            for d in experiment_data:
                if d.get("prop") == prop:
                    for run in d.get("runs", []):
                        for step in run:
                            ts = step.get("ntrain")
                            if ts and ts > 1:
                                train_sizes_set.add(ts)
            available_train_sizes = sorted(list(train_sizes_set))
            
            if not available_train_sizes:
                available_train_sizes = [100]

            min_train = available_train_sizes[0]
            max_train = available_train_sizes[-1]

            # 1. Fetch precalculated frames from cache
            slider_steps_list, frames_dict = precalculate_frames(
                experiment_data, prop, selected_kernel, reps, datasets, min_train, max_train, available_train_sizes
            )

            # Fix 1: Calculate global min/max for MAE and RMSE across ALL frames to lock the gradient range
            all_mae_vals = [v for f in frames_dict.values() for row in f['MAE']['z'] for v in row if v is not None]
            all_rmse_vals = [v for f in frames_dict.values() for row in f['RMSE']['z'] for v in row if v is not None]

            zmin_mae, zmax_mae = (min(all_mae_vals), max(all_mae_vals)) if all_mae_vals else (None, None)
            zmin_rmse, zmax_rmse = (min(all_rmse_vals), max(all_rmse_vals)) if all_rmse_vals else (None, None)

            # Initialize at maximum train size
            init_val = max_train
            init_data = frames_dict[init_val]
            init_colorscale = REAL_COLORSCALE if init_data['is_real'] else INTERPOLATED_COLORSCALE
            bold_datasets = [f"<b>{ds}</b>" for ds in datasets]

            # Fix 3: Increased horizontal_spacing to prevent overlapping between MAE colorbar and RMSE Y-axis
            fig = make_subplots(
                rows=1, cols=2, 
                subplot_titles=("<b>MAE</b>", "<b>RMSE</b>"), 
                horizontal_spacing=0.22
            )

            # Initial trace for MAE
            fig.add_trace(go.Heatmap(
                z=init_data['MAE']['z'], x=bold_datasets, y=reps, text=init_data['MAE']['text'],
                texttemplate="%{text}", colorscale=init_colorscale,
                zmin=zmin_mae, zmax=zmax_mae,
                colorbar=dict(title="MAE", x=0.41),
                hoverinfo="x+y+text"
            ), row=1, col=1)

            # Initial trace for RMSE
            fig.add_trace(go.Heatmap(
                z=init_data['RMSE']['z'], x=bold_datasets, y=reps, text=init_data['RMSE']['text'],
                texttemplate="%{text}", colorscale=init_colorscale,
                zmin=zmin_rmse, zmax=zmax_rmse,
                colorbar=dict(title="RMSE", x=1.02),
                hoverinfo="x+y+text"
            ), row=1, col=2)

            # 3. Build Client-Side Plotly Frames with locked gradient range
            frames = []
            for val in slider_steps_list:
                frame_data = frames_dict[val]
                c_scale = REAL_COLORSCALE if frame_data['is_real'] else INTERPOLATED_COLORSCALE
                
                frames.append(go.Frame(
                    name=str(val),
                    data=[
                        go.Heatmap(
                            z=frame_data['MAE']['z'], text=frame_data['MAE']['text'], 
                            colorscale=c_scale, zmin=zmin_mae, zmax=zmax_mae
                        ),
                        go.Heatmap(
                            z=frame_data['RMSE']['z'], text=frame_data['RMSE']['text'], 
                            colorscale=c_scale, zmin=zmin_rmse, zmax=zmax_rmse
                        )
                    ]
                ))

            fig.frames = frames

            # Configure Native Plotly Slider
            steps = []
            for val in slider_steps_list:
                step = dict(
                    method="animate",
                    args=[
                        [str(val)],
                        dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))
                    ],
                    label=str(val)
                )
                steps.append(step)

            active_index = slider_steps_list.index(init_val) if init_val in slider_steps_list else len(slider_steps_list) - 1

            sliders = [dict(
                active=active_index,
                currentvalue={"prefix": "Training Points: "},
                pad={"t": 40},
                steps=steps
            )]

            # Fix 3: Adjust subplot titles above X-axis tick labels
            for ann in fig['layout']['annotations']:
                ann['y'] = 1.15

            # Fix 2 & 3: Disable dragmode, expand top margin, fix axes range
            label_font = dict(family=FONT_FAMILY, color=DARK_GREY, size=12)
            fig.update_layout(
                dragmode=False,
                sliders=sliders,
                margin=dict(t=80, b=10, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family=FONT_FAMILY, color=DARK_GREY),
                height=300 + (len(reps) * 40)
            )

            # Fix 2: Disable zoom and panning on x/y axes
            fig.update_xaxes(side="top", tickfont=label_font, fixedrange=True)
            fig.update_yaxes(autorange="reversed", tickfont=dict(family=FONT_FAMILY, color=DARK_GREY, weight="bold"), fixedrange=True)

            # Fix 2: Render with modebar and scroll zoom disabled
            st.plotly_chart(
                fig, 
                width='stretch', 
                key=f"heatmap_{prop}", 
                config={'displayModeBar': False, 'scrollZoom': False}
            )
            
            st.markdown("<br><hr style='margin: 20px 0;'><br>", unsafe_allow_html=True)