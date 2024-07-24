import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt




def read_dataframe_from_file(file_path):
    """
    Reads a DataFrame from a file.

    Parameters:
    file_path (str): The path to the file from which the DataFrame will be read.

    Returns:
    pd.DataFrame: The DataFrame read from the file.
    """
    df = pd.read_csv(file_path)
    print(f"DataFrame read from {file_path}")
    return df




def plot_cumulative_distribution(df):
    """
    Plots the cumulative distribution of the DataFrame entries grouped by module number.

    Parameters:
    df (pd.DataFrame): The DataFrame with 'module' and 'timestamp' columns.
    """
    plt.figure(figsize=(10, 6))
    
    for module in df['module'].unique():
        module_df = df[df['module'] == module]
        module_df = module_df.sort_values(by='timestamp')
        cumulative_counts = module_df['timestamp'].expanding().count()
        
        plt.plot(module_df['timestamp'], cumulative_counts, label=f'Module {module}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cumulative Count')
    plt.title('Cumulative Distribution of Module Events Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()




def plot_timestamp_distribution(df, bin_size=10):
    """
    Plots the distribution of timestamps grouped by module number.

    Parameters:
    df (pd.DataFrame): The DataFrame with 'module' and 'timestamp' columns.
    bin_size (int): The size of the bins for the histogram.
    """
    plt.figure(figsize=(10, 6))
    
    for module in df['module'].unique():
        module_df = df[df['module'] == module]
        plt.hist(module_df['timestamp'], bins=bin_size, alpha=0.5, label=f'Module {module}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Timestamp Distribution of Module Events')
    plt.legend()
    plt.grid(True)
    plt.show()




def plot_histograms_separate_pads(df, bin_size=10):
    """
    Plots the distribution of timestamps grouped by module number in separate subplots.

    Parameters:
    df (pd.DataFrame): The DataFrame with 'module' and 'timestamp' columns.
    bin_size (int): The size of the bins for the histogram.
    """
    modules = df['module'].unique()
    num_modules = len(modules)
    fig, axes = plt.subplots(num_modules, 1, figsize=(10, 6 * num_modules), sharex=True)
    
    for i, module in enumerate(modules):
        module_df = df[df['module'] == module]
        axes[i].hist(module_df['timestamp'], bins=bin_size, alpha=0.5)
        axes[i].set_title(f'Module {module} Timestamp Distribution')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True)
    
    plt.xlabel('Time (seconds)')
    plt.tight_layout()
    plt.show()



def analyze_module_events(df):
    """
    Groups the DataFrame by module number, counts the number of entries for each module,
    and computes the total time as the difference between the maximum and minimum timestamps.

    Parameters:
    df (pd.DataFrame): The DataFrame with 'module' and 'timestamp' columns.

    Returns:
    pd.DataFrame: A DataFrame with counts of entries for each module.
    float: The total time (difference between maximum and minimum timestamps).
    """
    # Ensure the DataFrame has the required columns
    if 'module' not in df.columns or 'timestamp' not in df.columns:
        raise ValueError("DataFrame must contain 'module' and 'timestamp' columns.")

    # Group by module number and count the number of entries for each module
    module_counts = df.groupby('module').size().reset_index(name='count')

    # Compute the total time as the difference between the maximum and minimum timestamps
    total_time = df['timestamp'].max() - df['timestamp'].min()

    return module_counts, total_time
