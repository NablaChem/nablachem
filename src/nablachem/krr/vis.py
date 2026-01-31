import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import sys

st.set_page_config(page_title="Training details", layout="wide")


@st.cache_data
def load_data():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "archive.json"

    with open(filename, "r") as f:
        data = json.load(f)
    return data["hyperopt"], data.get("spectrum", {}), data["learning_curve"]


def process_hyperopt_data(hyperopt_data):
    processed_data = defaultdict(list)

    for entry in hyperopt_data:
        ntrain = entry["ntrain"]
        sigma = entry["sigma"]
        lambda_val = entry["lambda"]

        # Calculate medians for all available metrics
        val_rmse_median = np.median(entry["val_rmse"])
        val_mae_median = np.median(entry["val_mae"])
        train_rmse_median = np.median(entry["train_rmse"])
        train_mae_median = np.median(entry["train_mae"])

        # Calculate max of training and validation RMSE
        max_rmse_median = max(train_rmse_median, val_rmse_median)

        processed_data[ntrain].append(
            {
                "sigma": sigma,
                "lambda": lambda_val,
                "val_rmse_median": val_rmse_median,
                "val_mae_median": val_mae_median,
                "train_rmse_median": train_rmse_median,
                "train_mae_median": train_mae_median,
                "max_rmse_median": max_rmse_median,
                "log_sigma": np.log10(sigma),
                "log_lambda": np.log10(lambda_val),
                "log_val_rmse_median": np.log10(val_rmse_median),
                "log_val_mae_median": np.log10(val_mae_median),
                "log_train_rmse_median": np.log10(train_rmse_median),
                "log_train_mae_median": np.log10(train_mae_median),
                "log_max_rmse_median": np.log10(max_rmse_median),
            }
        )

    return processed_data


def create_heatmap_plot(
    data_for_ntrain, ntrain, metric_key, metric_display_name, vmin=None, vmax=None
):
    df = pd.DataFrame(data_for_ntrain)

    # Define metric mappings
    metric_to_median_key = {
        "val_rmse": "val_rmse_median",
        "val_mae": "val_mae_median",
        "train_rmse": "train_rmse_median",
        "train_mae": "train_mae_median",
        "max_rmse": "max_rmse_median",
    }

    metric_to_log_key = {
        "val_rmse": "log_val_rmse_median",
        "val_mae": "log_val_mae_median",
        "train_rmse": "log_train_rmse_median",
        "train_mae": "log_train_mae_median",
        "max_rmse": "log_max_rmse_median",
    }

    median_key = metric_to_median_key[metric_key]
    log_key = metric_to_log_key[metric_key]

    # Find the minimum point (only for validation metrics, not for training or max metrics)
    min_point = None
    show_best_point = metric_key.startswith("val_")
    if show_best_point:
        min_idx = df[median_key].idxmin()
        min_point = df.loc[min_idx]

    # Get unique values for lambda and sigma (original values, not log)
    unique_lambda = sorted(df["lambda"].unique())
    unique_sigma = sorted(df["sigma"].unique())

    # Create grid matrix
    grid_shape = (len(unique_sigma), len(unique_lambda))
    Z = np.full(grid_shape, np.nan)
    hover_text = np.full(grid_shape, "", dtype=object)

    # Fill the grid with actual calculated values
    for _, row in df.iterrows():
        sigma_idx = unique_sigma.index(row["sigma"])
        lambda_idx = unique_lambda.index(row["lambda"])
        Z[sigma_idx, lambda_idx] = row[log_key]
        hover_text[sigma_idx, lambda_idx] = (
            f"λ: {row['lambda']:.2e}<br>σ: {row['sigma']:.6f}<br>median {metric_display_name.lower()}: {row[median_key]:.3e}"
        )

    # Create heatmap using plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=Z,
            x=unique_lambda,
            y=unique_sigma,
            colorscale="viridis",
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(
                title=f"median {metric_display_name.lower()}",
                tickmode="array",
                tickvals=(
                    np.arange(np.ceil(vmin), np.floor(vmax) + 1)
                    if vmin is not None and vmax is not None
                    else None
                ),
                ticktext=(
                    [
                        f"{10**val:.2e}"
                        for val in np.arange(np.ceil(vmin), np.floor(vmax) + 1)
                    ]
                    if vmin is not None and vmax is not None
                    else None
                ),
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            hoverongaps=False,
        )
    )

    # Add marker for best point (only for validation metrics)
    if show_best_point and min_point is not None:
        fig.add_trace(
            go.Scatter(
                x=[min_point["lambda"]],
                y=[min_point["sigma"]],
                mode="markers",
                marker=dict(
                    color="red",
                    size=15,
                    symbol="circle-open",
                    line=dict(width=3, color="red"),
                ),
                name="Best Performance",
                hovertemplate=f"<b>Best Point</b><br>"
                + f'λ: {min_point["lambda"]:.2e}<br>'
                + f'σ: {min_point["sigma"]:.6f}<br>'
                + f"median {metric_display_name.lower()}: {min_point[median_key]:.3e}<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        xaxis=dict(
            title=dict(text="λ", font=dict(size=24)),
            type="log",
            tickformat=".0e",
            exponentformat="power",
            tickfont=dict(size=20),
            dtick=1,
        ),
        yaxis=dict(
            title=dict(text="σ", font=dict(size=24)),
            type="log",
            tickformat=".0e",
            exponentformat="power",
            tickfont=dict(size=20),
            dtick=1,
        ),
        width=800,
        height=400,
        showlegend=show_best_point,
    )

    return fig, min_point


def create_eigenvalue_plot(spectrum_data, ntrain):
    if str(ntrain) not in spectrum_data:
        return None

    eigenvalues = np.array(spectrum_data[str(ntrain)])
    sorted_eigenvalues = np.sort(eigenvalues)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(sorted_eigenvalues)),
            y=sorted_eigenvalues,
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=4),
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Kernel matrix eigenspectrum",
        xaxis=dict(
            title=dict(text="Index (sorted)", font=dict(size=24)),
            tickfont=dict(size=20),
        ),
        yaxis=dict(
            title=dict(text="Eigenvalue", font=dict(size=24)),
            type="log",
            tickfont=dict(size=20),
            tickformat=".0e",
            exponentformat="power",
            dtick=1,
        ),
        width=800,
        height=400,
        showlegend=False,
    )

    return fig


def create_learning_curve_plot(learning_curve_data):
    """Create learning curve plot matching the original matplotlib style (without theoretical limit)"""
    # Extract data
    training_sizes = []
    val_rmse = []
    test_rmse = []
    val_mae = []
    test_mae = []
    nullmodel_rmse = None

    for entry in learning_curve_data:
        ntrain = entry["ntrain"]
        if ntrain == 1:  # nullmodel
            nullmodel_rmse = entry["test_rmse"]
        else:
            training_sizes.append(ntrain)
            val_rmse.append(entry["val_rmse"])
            test_rmse.append(entry["test_rmse"])
            val_mae.append(entry["val_mae"])
            test_mae.append(entry["test_mae"])

    fig = go.Figure()

    # Add RMSE traces
    fig.add_trace(
        go.Scatter(
            x=training_sizes,
            y=val_rmse,
            mode="lines+markers",
            name="val RMSE",
            line=dict(color="#1f77b4", width=1),
            marker=dict(symbol="circle", size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=training_sizes,
            y=test_rmse,
            mode="lines+markers",
            name="test RMSE",
            line=dict(color="#ff7f0e", width=1),
            marker=dict(symbol="square", size=6),
        )
    )

    # Add MAE traces
    fig.add_trace(
        go.Scatter(
            x=training_sizes,
            y=val_mae,
            mode="lines",
            name="val MAE",
            line=dict(color="#1f77b4", width=1),
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=training_sizes,
            y=test_mae,
            mode="lines",
            name="test MAE",
            line=dict(color="#ff7f0e", width=1),
            showlegend=True,
        )
    )

    # Add nullmodel horizontal line
    if nullmodel_rmse and training_sizes:
        min_training_size = min(training_sizes)
        max_training_size = max(training_sizes)
        fig.add_shape(
            type="line",
            x0=min_training_size * 0.9,
            x1=max_training_size * 1.2,
            y0=nullmodel_rmse,
            y1=nullmodel_rmse,
            line=dict(color="gray", width=1, dash="dash"),
            opacity=0.5,
        )
        fig.add_annotation(
            x=max_training_size * 0.9,
            y=nullmodel_rmse * 1.05,
            text="nullmodel",
            showarrow=False,
            font=dict(color="gray", size=10),
        )

    # Set layout to match matplotlib style
    if training_sizes and val_rmse and test_rmse and val_mae and test_mae:
        all_errors = val_rmse + test_rmse + val_mae + test_mae
        min_error = min(all_errors)
        max_error = max(all_errors)
        max_training_size = max(training_sizes)
        min_training_size = min(training_sizes)

        # Tight bounds with minimal padding
        y_lower_bound = min_error * 0.9
        y_upper_bound = (
            max(nullmodel_rmse, max_error) * 1.05
            if nullmodel_rmse
            else max_error * 1.05
        )

        fig.update_layout(
            title="Learning Curve",
            xaxis=dict(
                title=dict(text="Training points", font=dict(size=24)),
                type="log",
                range=[
                    np.log10(min_training_size * 0.9),
                    np.log10(max_training_size * 1.2),
                ],
                tickfont=dict(size=20),
                tickmode="array",
                tickvals=training_sizes,
                ticktext=[str(int(n)) for n in training_sizes],
                showline=True,
                linewidth=1,
                linecolor="black",
            ),
            yaxis=dict(
                title=dict(text="RMSE/MAE", font=dict(size=24)),
                type="log",
                showgrid=True,
                dtick=1,
                gridcolor="#555",
                ticks="inside",
                minor={"dtick": 0, "showgrid": True, "ticks": "inside"},
                minorloglabels="none",
                range=[np.log10(y_lower_bound), np.log10(y_upper_bound)],
                tickfont=dict(size=20),
                # tickmode="array",
                # tickvals=([10**i for i in range(-6, 2)] +  # Major ticks: 10^-6 to 10^1
                #         [j * 10**i for i in range(-6, 1) for j in [2,3,4,5,6,7,8,9]]),  # Minor ticks
                # ticktext=([f"10<sup>{i}</sup>" for i in range(-6, 2)] +  # Labels for major ticks
                #         ["" for i in range(-6, 1) for j in [2,3,4,5,6,7,8,9]]),  # No labels for minor ticks
                # ticks="inside",
                showline=True,
                linewidth=1,
                linecolor="black",
            ),
            legend=dict(x=0.02, y=0.02, bgcolor="rgba(255,255,255,0.8)"),
            width=800,
            height=500,
            showlegend=True,
        )

    return fig


def main():
    st.markdown("Interactive visualization of hyperopt results from archive.json")

    # Load and process data
    hyperopt_data, spectrum_data, learning_curve_data = load_data()
    processed_data = process_hyperopt_data(hyperopt_data)

    # Get unique ntrain values and sort them
    ntrain_values = sorted(processed_data.keys())

    # Create tabs with learning curve as the leftmost tab
    tab_names = ["Learning Curve"] + [f"ntrain = {ntrain}" for ntrain in ntrain_values]
    tabs = st.tabs(tab_names)

    # Learning Curve Tab (leftmost)
    with tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            # Create and display learning curve plot
            lc_fig = create_learning_curve_plot(learning_curve_data)
            st.plotly_chart(lc_fig, use_container_width=True)

        with col2:
            # Create learning curve data table
            st.markdown("### Learning Curve Data")
            lc_df = pd.DataFrame(learning_curve_data)

            # Format the data for display
            display_lc_df = lc_df.copy()
            display_lc_df["sigma"] = display_lc_df["hyperparameters"].apply(
                lambda x: (
                    "inf"
                    if x.get("sigma") == float("inf")
                    else f"{x.get('sigma', 0):.3e}"
                )
            )
            display_lc_df["lambda"] = display_lc_df["hyperparameters"].apply(
                lambda x: (
                    "-"
                    if x.get("sigma") == float("inf")
                    else f"{x.get('lambda', 0):.3e}"
                )
            )

            # Select and rename columns for display
            display_lc_df = display_lc_df[
                [
                    "ntrain",
                    "val_rmse",
                    "test_rmse",
                    "val_mae",
                    "test_mae",
                    "sigma",
                    "lambda",
                ]
            ]
            display_lc_df.columns = [
                "ntrain",
                "val_rmse",
                "test_rmse",
                "val_mae",
                "test_mae",
                "σ",
                "λ",
            ]

            # Format numerical columns
            for col in ["val_rmse", "test_rmse", "val_mae", "test_mae"]:
                display_lc_df[col] = display_lc_df[col].apply(lambda x: f"{x:.4f}")

            st.dataframe(display_lc_df, use_container_width=True)

    # Hyperopt tabs (offset by 1 due to learning curve tab)
    for i, ntrain in enumerate(ntrain_values):
        with tabs[i + 1]:
            data_for_ntrain = processed_data[ntrain]

            # Define metric options for dropdown
            metric_options = {
                "Validation RMSE": "val_rmse",
                "Validation MAE": "val_mae",
                "Training RMSE": "train_rmse",
                "Training MAE": "train_mae",
                "Max(Training RMSE, Validation RMSE)": "max_rmse",
            }

            # Create side-by-side columns for plots
            col1, col2 = st.columns(2)

            with col1:
                # Add dropdown for metric selection
                selected_metric_display = st.selectbox(
                    "Select metric to display:",
                    options=list(metric_options.keys()),
                    index=0,  # Default to "Validation RMSE"
                    key=f"metric_selector_{ntrain}",
                )
                selected_metric_key = metric_options[selected_metric_display]

                # Calculate color range for the selected metric
                df = pd.DataFrame(data_for_ntrain)
                metric_to_log_key = {
                    "val_rmse": "log_val_rmse_median",
                    "val_mae": "log_val_mae_median",
                    "train_rmse": "log_train_rmse_median",
                    "train_mae": "log_train_mae_median",
                    "max_rmse": "log_max_rmse_median",
                }
                log_key = metric_to_log_key[selected_metric_key]

                if selected_metric_key.startswith("train_"):
                    # For training metrics: use full data domain
                    local_vmin = df[log_key].min()
                    local_vmax = df[log_key].max()
                else:
                    # For validation metrics and max_rmse: use current logic (90th percentile)
                    local_vmin = df[log_key].min()
                    local_vmax = df[log_key].quantile(0.90)

                # Create and display hyperparameter optimization plot
                fig, min_point = create_heatmap_plot(
                    data_for_ntrain,
                    ntrain,
                    selected_metric_key,
                    selected_metric_display,
                    local_vmin,
                    local_vmax,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Create and display eigenvalue plot
                eigenvalue_fig = create_eigenvalue_plot(spectrum_data, ntrain)
                if eigenvalue_fig:
                    st.plotly_chart(eigenvalue_fig, use_container_width=True)
                else:
                    st.info(f"No spectrum data available for ntrain={ntrain}")

            # Show raw hyperopt data
            df = pd.DataFrame(data_for_ntrain)
            metric_to_median_key = {
                "val_rmse": "val_rmse_median",
                "val_mae": "val_mae_median",
                "train_rmse": "train_rmse_median",
                "train_mae": "train_mae_median",
                "max_rmse": "max_rmse_median",
            }
            median_key = metric_to_median_key[selected_metric_key]

            display_df = df[["sigma", "lambda", median_key]].copy()
            display_df["sigma"] = display_df["sigma"].apply(lambda x: f"{x:.3e}")
            display_df["lambda"] = display_df["lambda"].apply(lambda x: f"{x:.3e}")
            display_df.columns = ["σ", "λ", f"Median {selected_metric_display.lower()}"]
            st.dataframe(display_df, use_container_width=True)


if __name__ == "__main__":
    main()
