import os
import matplotlib.pyplot as plt
import csv
from absl import app
from absl import flags

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None,
    "Directory to store parameter values per query.")
flags.mark_flag_as_required("output_dir")
_PQO_LATENCY_FILE = flags.DEFINE_string(
    "pqo_latency_file", None,
    "CSV file regarding pqo latency result.")
flags.mark_flag_as_required("pqo_latency_file")
_QUERY_ID = flags.DEFINE_string("query_id", None, "Name of query to train on")
flags.mark_flag_as_required("query_id")
_TRAINING_SIZE = flags.DEFINE_string("training_size", None, "Training set length")
flags.mark_flag_as_required("training_size")


def load_results_from_csv(pqo_latency_file):
    results = []
    with open(pqo_latency_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            results.append({
                'query': row['query'],
                'frequency': int(row['frequency']),
                'hinted_latency': float(row['hinted_latency']),
                'default_latency': float(row['default_latency']),
                'hinted_cost': float(row['hinted_cost']),
                'default_cost': float(row['default_cost']),
                'plan_content': row['plan_content']
            })
    return results

def expand_results_by_frequency(results):
    expanded_results = []
    for result in results:
        expanded_results.extend([result] * result['frequency'])  # Expand based on frequency
    return expanded_results


def sort_results_by_default_latency(results):
    return sorted(results, key=lambda x: x['default_latency'])


def plot_latency(output_dir, query_id, training_size, hinted_latency, default_latency, hinted_cost, default_cost, title, file_suffix, sort=False):
    x_values = range(len(hinted_latency))  # X-axis values for plotting
    
    if sort:
        # Combined plot for latency and cost when sorted=True
        fig, ax1 = plt.subplots(figsize=(10, 6))
        # Plot latency on the first y-axis
        ax1.plot(x_values, hinted_latency, color='blue', marker='o', label='Robust Plan Latency')
        ax1.plot(x_values, default_latency, color='orange', marker='o', label='Original Latency')
        ax1.set_xlabel('Query Index')
        ax1.set_ylabel('Latency (ms)')
        title = f"{query_id} Hinted vs Default Latency (Sorted by Default Latency, All Data Expanded)"
        ax1.set_title(title)

        # Add a second y-axis for cost
        ax2 = ax1.twinx()
        ax2.plot(x_values, hinted_cost, color='darkblue', marker='x', label='Robust Plan Cost')
        ax2.plot(x_values, default_cost, color='red', marker='x', label='Original Cost')
        ax2.set_ylabel('Cost')

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        # Save combined figure - {query_id}_{training_size}_{file_suffix}
        figure_name = f'{output_dir}/{query_id}_{training_size}_{file_suffix}_sorted_latency_cost.png'
        plt.savefig(figure_name)
        plt.close(fig)
        print(f"Plot saved: {figure_name}")

    else:
        # Generate latency-only plot when sorted=False
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(x_values, hinted_latency, color='blue', marker='o', label='Robust Plan Latency')
        ax1.plot(x_values, default_latency, color='orange', marker='o', label='Original Latency')
        ax1.set_xlabel('Query Index')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title(f"{query_id} Hinted vs Default Latency (All Data Expanded)")
        ax1.legend(loc='upper left')

        # Save latency-only figure - {query_id}_{training_size}_{file_suffix}
        latency_figure_name = f'{output_dir}/{query_id}_{training_size}_{file_suffix}_latency.png'
        plt.savefig(latency_figure_name)
        plt.close(fig)
        print(f"Latency plot saved: {latency_figure_name}")

        # Generate cost-only plot when sorted=False
        fig, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(x_values, hinted_cost, color='darkblue', marker='x', linestyle='--', label='Robust Plan Cost')
        ax2.plot(x_values, default_cost, color='red', marker='x', linestyle='--', label='Original Cost')
        ax2.set_xlabel('Query Index')
        ax2.set_ylabel('Cost')
        ax2.set_title(f"{query_id} Hinted vs Default Cost (All Data Expanded)")
        ax2.legend(loc='upper left')

        # Save cost-only figure - {query_id}_{training_size}_{file_suffix}
        cost_figure_name = f'{output_dir}/{query_id}_{training_size}_{file_suffix}_cost.png'
        plt.savefig(cost_figure_name)
        plt.close(fig)
        print(f"Cost plot saved: {cost_figure_name}")
    

# Plot chunk (use original distinct result: result)
def plot_static_chunks_sorted(output_dir, query_id, training_size, results, chunk_size=100, split_threshold=0.2):
    sorted_results = sort_results_by_default_latency(results)

    for i in range(0, len(sorted_results), chunk_size):
        chunk = sorted_results[i:i + chunk_size]
        hinted_latency = [result['hinted_latency'] for result in chunk]
        default_latency = [result['default_latency'] for result in chunk]
        hinted_cost = [result['hinted_cost'] for result in chunk]
        default_cost = [result['default_cost'] for result in chunk]

        # Calculate percentage change in default latency
        max_latency = max(default_latency)
        min_latency = min(default_latency)
        percentage_change = (max_latency - min_latency) / min_latency if min_latency > 0 else 1

        if len(chunk) >= 50 and percentage_change > split_threshold:
            # Split the chunk into two and plot separately if percentage change is large
            mid_point = len(chunk) // 2
            plot_latency(
                output_dir,
                query_id,
                training_size,
                hinted_latency[:mid_point],
                default_latency[:mid_point],
                hinted_cost[:mid_point],
                default_cost[:mid_point],
                title=f'{query_id} Hinted vs Default Latency (Sorted, Chunk {i // chunk_size + 1}_Part_1)',
                file_suffix=f'chunk_{i // chunk_size + 1}_part_1_sorted_latency',
                sort=True
            )

            plot_latency(
                output_dir,
                query_id,
                training_size,
                hinted_latency[mid_point:],
                default_latency[mid_point:],
                hinted_cost[mid_point:],
                default_cost[mid_point:],
                title=f'{query_id} Hinted vs Default Latency (Sorted, Chunk {i // chunk_size + 1}_Part_2)',
                file_suffix=f'chunk_{i // chunk_size + 1}_part_2_sorted_latency',
                sort=True
            )
        else:
            # Plot as a single chunk if percentage change is small
            plot_latency(
                output_dir,
                query_id,
                training_size,
                hinted_latency,
                default_latency,
                hinted_cost,
                default_cost,
                title=f'{query_id} Hinted vs Default Latency (Sorted by Default Latency, Chunk {i // chunk_size + 1})',
                file_suffix=f'chunk_{i // chunk_size + 1}_sorted_latency',
                sort=True
            )



def main(unused_arg):
    output_dir = _OUTPUT_DIR.value
    pqo_latency_file = _PQO_LATENCY_FILE.value
    query_id = _QUERY_ID.value
    training_size = _TRAINING_SIZE.value
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results from CSV
    results = load_results_from_csv(pqo_latency_file)
    
    # Expanded result
    expanded_results = expand_results_by_frequency(results)
    hinted_latency = [result['hinted_latency'] for result in expanded_results]
    default_latency = [result['default_latency'] for result in expanded_results]
    hinted_cost = [result['hinted_cost'] for result in expanded_results]
    default_cost = [result['default_cost'] for result in expanded_results]
    
    # Plot the full result
    plot_latency(
        output_dir,
        query_id,
        training_size,
        hinted_latency,
        default_latency,
        hinted_cost,
        default_cost,
        title=f'{query_id} Hinted vs Default Latency - ALL expanded result',
        file_suffix=f'full_latency'
    )
    
    # Plot the full result
    expanded_results = sort_results_by_default_latency(expanded_results)
    hinted_latency = [result['hinted_latency'] for result in expanded_results]
    default_latency = [result['default_latency'] for result in expanded_results]
    hinted_cost = [result['hinted_cost'] for result in expanded_results]
    default_cost = [result['default_cost'] for result in expanded_results]
    
    plot_latency(
        output_dir,
        query_id,
        training_size,
        hinted_latency,
        default_latency,
        hinted_cost,
        default_cost,
        title=f'{query_id} Hinted vs Default Latency (Sorted by Default Latency) - ALL expanded result',
        file_suffix=f'full_sorted_latency',
        sort=True
    )
    
    
    # Plot sorted static 100-row chunks without expanding
    plot_static_chunks_sorted(output_dir, query_id, training_size, results, chunk_size=100)


if __name__ == "__main__":
    app.run(main)
