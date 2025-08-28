import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, entropy, wasserstein_distance
import re
from collections import Counter
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

#1

def ordered_SF(sf):
    """
    Normalises a sum formula (SF) by ordering elements alphabetically and consolidating their counts.

    arguments: sf (str): A sum formula string (e.g., "C4H8O2").

    it returns a str: The normalized (ordered) formula string (e.g., "C4H8O2" -> "C4H8O2").

    """
    # Extract elements and their counts
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", sf)
    
    # Consolidate the counts
    element_counts = Counter()
    for elem, count in matches:
        element_counts[elem] += int(count) if count else 1

    # Build the normalised formula in alphabetical order
    normalised_formula = ""
    for elem in sorted(element_counts.keys()):  # Alphabetical order
        normalised_formula += elem
        if element_counts[elem] > 1:  # Add count if greater than 1
            normalised_formula += str(element_counts[elem])
    
    return normalised_formula

#2
def plot_cdfs(data_frame, distribution_one, distribution_two, dataset_name="Dataset"):
    """
    Plots the cumulative distribution functions (CDFs) for two distributions.

    Arguments:
    - data_frame (pd.DataFrame): The input DataFrame containing the distributions.
    - distribution_one (str): The column name for the first distribution.
    - distribution_two (str): The column name for the second distribution.
    - dataset_name (str): A label for the dataset being plotted.
    """
    # Ordering by the first distribution
    ordering = np.argsort(data_frame[distribution_one].values)
    
    # Apply the same order to both distributions
    ordered_dist_one = data_frame[distribution_one].values[ordering]
    ordered_dist_two = data_frame[distribution_two].values[ordering]
    
    # Calculate CDFs
    cumsum_one = np.cumsum(ordered_dist_one) / np.sum(ordered_dist_one)
    cumsum_two = np.cumsum(ordered_dist_two) / np.sum(ordered_dist_two)
    
    # Generate cumulative indices
    x_values = np.arange(1, len(ordered_dist_one) + 1)  # Use real cumulative indices
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.semilogy(x_values, cumsum_one, label=f"CDF {distribution_one} ({dataset_name})", color="blue", linewidth=2)
    plt.semilogy(x_values, cumsum_two, label=f"CDF {distribution_two} ({dataset_name})", color="red", linewidth=3)
    # np.abs(cumsum_one - cumsum_two).max()

    plt.title(f"CDF Comparison for {dataset_name}", fontsize=14)
    plt.xlabel("Cumulative SF", fontsize=12)
    plt.ylabel("Log Scale Frequency", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

#3
def calculate_metrics(data_frame, distribution_one, distribution_two):
    """
    Calculates statistical metrics between two distributions:
    - KS statistic
    - Area under the CDFs
    - Percentage difference of the area between the CDFs relative to the max curve
    - Kullback-Leibler (KL) divergences
    
        data_frame (pd.DataFrame): The input DataFrame containing the distributions.
        distribution_one (str): Column name for the first distribution.
        distribution_two (str): Column name for the second distribution.

    it returns a dataframe
    """
    # Extract distributions
    dist_one = data_frame[distribution_one].values
    dist_two = data_frame[distribution_two].values

    # Normalize distributions
    p = dist_one / dist_one.sum()
    q = dist_two / dist_two.sum()

    # Replace zeros to avoid log issues in KL divergence
    p += 1e-10
    q += 1e-10

    # Calculate KL divergence
    kl_pq = entropy(p, q)  # KL(P || Q)

    # Calculate CDFs
    cumsum_one = np.cumsum(p)
    cumsum_two = np.cumsum(q)

    # Calculate KS statistic manually (maximum absolute difference between CDFs)
    ks_stat = np.max(np.abs(cumsum_one - cumsum_two))

    # Calculate differences between CDFs
    cdf_diff = np.abs(cumsum_one - cumsum_two)

    # Area under the curves
    area_one = np.trapz(cumsum_one, dx=1)
    area_two = np.trapz(cumsum_two, dx=1)

    # Area between the CDFs
    area_between_cdfs = np.trapz(cdf_diff, dx=1)

    # Percentage difference of the area relative to the maximum area
    max_area = max(area_one, area_two)
    percentage_diff_area = (area_between_cdfs / max_area) * 100

    # Create results dictionary
    results = {
        "KS Statistic": ks_stat,
        "Area under Distribution One": area_one,
        "Area under Distribution Two": area_two,
        "Area between the CDFs": area_between_cdfs,
        "Percentage Area Difference (relative to max curve)": percentage_diff_area,
        "KL Divergence (P || Q)": kl_pq,
    }

    # Convert to DataFrame
    results_df = pd.DataFrame(results.items(), columns=["Metric", "Value"])

    # Print the results
    print("Calculated Metrics:")
    print(results_df.to_string(index=False))

    return results_df


#4

def bootstrap_metrics(data_frame, distribution_one, distribution_two, n_bootstrap, alpha):
    """
    This function performs bootstrapping to compute confidence intervals and mean values
    for Kolmogorov-Smirnov (KS) and Kullback-Leibler (KL) divergences.
    """
    # Lists to store bootstrap results for metrics
    ks_values = []
    kl_pq_values = []

    # Perform bootstrap
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled_data = data_frame.sample(frac=1.0, replace=True, random_state=None)
        resampled_p = resampled_data[distribution_one] / resampled_data[distribution_one].sum()
        resampled_q = resampled_data[distribution_two] / resampled_data[distribution_two].sum()

        # Avoid issues with log(0)
        resampled_p += 1e-10
        resampled_q += 1e-10

        # Calculate metrics
        # KS Statistic manually (maximum absolute difference between CDFs)
        cumsum_p = np.cumsum(resampled_p)
        cumsum_q = np.cumsum(resampled_q)
        ks_stat = np.max(np.abs(cumsum_p - cumsum_q))

        kl_pq = entropy(resampled_p, resampled_q)  # KL(P || Q)

        # Store results
        ks_values.append(ks_stat)
        kl_pq_values.append(kl_pq)

    # Calculate mean and confidence intervals for each metric
    ks_mean = np.mean(ks_values)
    kl_pq_mean = np.mean(kl_pq_values)

    ks_ci = np.percentile(ks_values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    kl_pq_ci = np.percentile(kl_pq_values, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Create a DataFrame to display results
    results_ci = pd.DataFrame({
        "Metric": ["KS", "KL_PQ"],
        "Value": [ks_mean, kl_pq_mean],
        "Confidence Interval": [
            f"[{ks_ci[0]:.3f}, {ks_ci[1]:.3f}]",
            f"[{kl_pq_ci[0]:.3f}, {kl_pq_ci[1]:.3f}]"
        ]
    })

    # Display the results
    print(results_ci.to_string(index=False))
    return results_ci


#5
    
def random_sampling_subset(data_frame, distribution_one, distribution_two, sample_sizes, n_bootstrap):

    """
    This function performs random sampling on a dataset to calculate the Kolmogorov-Smirnov (KS) 
    and Kullback-Leibler (KL) divergences for different sample fractions. 

    It prints and returns a table summarising the results.

    Arguments:
    - data_frame: pd.DataFrame. The input DataFrame containing the distributions.
    - distribution_one: str. Column name for the first distribution.
    - distribution_two: str. Column name for the second distribution.
    - sample_sizes: list. Fractions of the dataset to sample.
    - n_bootstrap: int. Number of bootstrap iterations.
    """

    results = []  # To save the results
    for frac in sample_sizes:
        # Creating random sampling
        samples = data_frame.sample(frac=frac, replace=False)

        # Calculate metrics
        ks_stat, p_value = ks_2samp(samples[distribution_one], samples[distribution_two])
        p = samples[distribution_one].values / samples[distribution_one].sum()
        q = samples[distribution_two].values / samples[distribution_two].sum()
        
        # Avoid issues with log(0)
        p += 1e-10
        q += 1e-10
        
        kl_pq = entropy(p, q)  # KL(P || Q)
        kl_qp = entropy(q, p)  # KL(Q || P)

        # Append results
        results.append({
            "Sample Fraction": frac,
            "KS": ks_stat,
            "KL_PQ": kl_pq,
            "KL_QP": kl_qp,
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Print the table
    print(results_df.to_string(index=False))  # Print the table in a clean format

    return results_df

#6
def calculate_relative_weights(data_frame, distribution_one, distribution_two):
    """
    Calculates relative weights for the provided DataFrame based on the specified columns.

    Parameters:
    - data_frame: pd.DataFrame. DataFrame containing the frequencies for space and database


    Returns:
    - pd.DataFrame: Updated DataFrame with three new columns:
        - 'Weight_' + distribution_one: Normalized weight based on the first distribution.
        - 'Weight_' + distribution_two: Normalized weight based on the second distribution.
        - 'Weight_combined': Normalized weight based on the combined sum of both distributions.
    """
    # Check if the columns exist
    if distribution_one not in data_frame.columns or distribution_two not in data_frame.columns:
        raise ValueError(f"Columns '{distribution_one}' and/or '{distribution_two}' are not in the DataFrame.")

    # Normalize frequencies to compute weights
    total_space = data_frame[distribution_one].sum()
    total_ani = data_frame[distribution_two].sum()
    total_combined = (data_frame[distribution_one] + data_frame[distribution_two]).sum()

    if total_space > 0:
        data_frame['Weight_' + distribution_one] = data_frame[distribution_one] / total_space
    else:
        data_frame['Weight_' + distribution_one] = 0  # Handle edge case

    if total_ani > 0:
        data_frame['Weight_' + distribution_two] = data_frame[distribution_two] / total_ani
    else:
        data_frame['Weight_' + distribution_two] = 0  # Handle edge case

    if total_combined > 0:
        data_frame['Weight_combined'] = (data_frame[distribution_one] + data_frame[distribution_two]) / total_combined
    else:
        data_frame['Weight_combined'] = 0  # Handle edge case

    return data_frame