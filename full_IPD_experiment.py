from IPDv6 import *
from visualizations import *
import glob


CONDITIONS = [
    "repeated_only",
    "competition_only",
    "super_additive"
]


def load_final_state_from_file(filepath: str) -> AddableDict:
    try:
        with open(filepath, 'r') as f:
         data = json.load(f)
        # Ensure the loaded data is wrapped in AddableDict if necessary for your setup
        # If AddableDict is just a dict subclass, this might be enough:
        return AddableDict(data)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

results = {}


for condition in CONDITIONS:
    file_paths = run_full_experiment(condition=condition, replications=5, model="mistral-small")
    #file_paths = glob.glob(f"plots/**/results_{condition}_*_qwen3.json", recursive=True)

    all_final_states = []
    for fp in file_paths:
        fs = load_final_state_from_file(fp)
        if fs:
            all_final_states.append(fs)

    if all_final_states:
        visualize_data_replications(all_final_states)
        print(f"\n {condition} \n")
        results[condition] = calculate_overall_mean_action_a_rate(all_final_states)
    else:
        print("No data loaded for visualization.")