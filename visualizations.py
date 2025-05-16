
from langchain_core.runnables import AddableDict
from typing import Dict, List, Tuple  # Added Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats  # For confidence intervals

PLOT_POINTS = 20  # Must be less or equal than min rounds

plt.rcParams.update({
    'font.size': 33,           # Default font size
    'axes.titlesize': 33,      # Title font size
    'axes.labelsize': 33,      # Axis label font size
    'xtick.labelsize': 28,     # X-axis tick label size
    'ytick.labelsize': 28,     # Y-axis tick label size
    'legend.fontsize': 33,     # Legend font size
    'figure.titlesize': 33     # Figure title size
})

def calculate_mean_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculates mean, and lower/upper bounds of the confidence interval."""
    if not data or all(np.isnan(d) for d in data):  # Handle empty or all-NaN data
        return np.nan, np.nan, np.nan

    data = [d for d in data if not np.isnan(d)]  # Remove NaNs for calculation
    if not data:  # If only NaNs were present
        return np.nan, np.nan, np.nan

    mean = np.mean(data)
    if len(data) < 2:  # Not enough data for CI
        return mean, mean, mean  # CI is undefined, return mean for bounds

    sem = stats.sem(data)
    if sem == 0 or np.isnan(sem):  # If SEM is zero (all values identical) or NaN
        return mean, mean, mean

    # Use t-distribution for CI, as sample size (number of replications) might be small
    ci_lower, ci_upper = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=sem)
    return mean, ci_lower, ci_upper


def visualize_tournament_results_replications(final_states: List[AddableDict]):
    """Generate visualizations of tournament results averaged over replications."""
    if not final_states:
        print("No final states provided for tournament visualization.")
        return

    experiment_condition = final_states[0].get('experiment_condition', 'default_exp')
    num_replications = len(final_states)

    # --- Plot 1: Cooperation rates over time ---
    fig_action_a, ax_action_a = plt.subplots(figsize=(12, 7))

    max_rounds_overall = 0
    for fs in final_states:
        max_rounds_overall = max(max_rounds_overall, fs.get("round_number", 0))

    # Stores lists of player-averaged action_a_rates for each round, across replications
    # all_replications_action_a_rates_by_round[round_idx] = [avg_rate_repl1, avg_rate_repl2, ...]
    all_replications_action_a_rates_by_round = [[] for _ in range(max_rounds_overall)]

    for fs in final_states:
        fs_round_num = fs.get("round_number", 0)
        # Collect action_a_rates for all players in this fs for each round
        action_a_rates_this_fs_by_round = [[] for _ in range(fs_round_num)]
        for _, player_stats in fs.get('players', {}).items():
            for r_num, rate in enumerate(player_stats.get('action_a_rate_history', [])):
                if r_num < fs_round_num:
                    action_a_rates_this_fs_by_round[r_num].append(rate)

        # Average player rates within this fs for each round
        for r_num in range(fs_round_num):
            if action_a_rates_this_fs_by_round[r_num]:
                all_replications_action_a_rates_by_round[r_num].append(np.mean(action_a_rates_this_fs_by_round[r_num]))
            # If no player data for this round in this fs, it remains empty, handled by calculate_mean_ci

    mean_history_repl = []
    ci_lower_history_repl = []
    ci_upper_history_repl = []

    for round_idx in range(max_rounds_overall):
        mean, ci_low, ci_up = calculate_mean_ci(all_replications_action_a_rates_by_round[round_idx])
        mean_history_repl.append(mean)
        ci_lower_history_repl.append(ci_low)
        ci_upper_history_repl.append(ci_up)

    # Smoothing
    x_values_smooth, smooth_average_repl, smooth_ci_lower_repl, smooth_ci_upper_repl = [], [], [], []
    n_smooth = max(1, int(len(mean_history_repl) / PLOT_POINTS))

    for i in range(0, len(mean_history_repl), n_smooth):
        chunk_means = [m for m in mean_history_repl[i:i + n_smooth] if not np.isnan(m)]
        chunk_lowers = [l for l in ci_lower_history_repl[i:i + n_smooth] if not np.isnan(l)]
        chunk_uppers = [u for u in ci_upper_history_repl[i:i + n_smooth] if not np.isnan(u)]

        if chunk_means:
            x_values_smooth.append(i)
            smooth_average_repl.append(np.mean(chunk_means))
            # For CI of smoothed data, it's an approximation to average the CI bounds.
            # A more rigorous approach involves smoothing the raw data from each replication first.
            if chunk_lowers:
                smooth_ci_lower_repl.append(np.mean(chunk_lowers))
            else:
                smooth_ci_lower_repl.append(np.nan)  # Propagate NaN if no valid CI bounds
            if chunk_uppers:
                smooth_ci_upper_repl.append(np.mean(chunk_uppers))
            else:
                smooth_ci_upper_repl.append(np.nan)  # Propagate NaN

    ax_action_a.plot(x_values_smooth, smooth_average_repl, marker='o')
    # Filter out NaNs before fill_between
    valid_indices = [j for j, (l, u) in enumerate(zip(smooth_ci_lower_repl, smooth_ci_upper_repl)) if
                     not (np.isnan(l) or np.isnan(u))]
    if valid_indices:
        ax_action_a.fill_between(
            [x_values_smooth[j] for j in valid_indices],
            [smooth_ci_lower_repl[j] for j in valid_indices],
            [smooth_ci_upper_repl[j] for j in valid_indices],
            alpha=0.2
        )
    plt.axhline(y=50, color='red', linestyle=':', linewidth=2)
    #ax_action_a.set_title(f'Cooperation rate evolution (Avg over {num_replications} replications)')
    ax_action_a.set_xlabel('Played rounds number')
    ax_action_a.set_ylabel('Cooperation Rate (%)')
    ax_action_a.set_ylim(0, 100)
    ax_action_a.legend()
    ax_action_a.grid(True)
    plt.tight_layout()
    plt.savefig(f'tournament_action_a_rate_replications_{experiment_condition}.png')
    plt.show()

    # --- Plot 2: First interaction cooperation rates ---
    fig_coop, ax_coop = plt.subplots(figsize=(12, 6))
    num_plot_points_coop = PLOT_POINTS

    for group_type in ['intergroup']:
        # all_trends_for_type[replication_idx] = [trend_point_1, trend_point_2, ...]
        all_trends_for_type = []
        max_trend_len = 0

        for fs in final_states:
            coop_data = []
            inter_data = fs.get('first_interaction_coop_intergroup', [])
            intra_data = fs.get('first_interaction_coop_intragroup', [])
            i = j = 0
            while i < len(inter_data):
                if j < len(intra_data):
                    coop_data.extend([intra_data[j], intra_data[j+1]])
                    j += 2
                coop_data.extend([inter_data[i], inter_data[i+1]])
                i += 2
            if not coop_data:
                all_trends_for_type.append([])  # Add empty list if no data for this rep
                continue

            current_trend = []
            interval = max(2, int(len(coop_data) / num_plot_points_coop))
            coop_count = 0
            interactions_processed = 0
            for i, coop_event in enumerate(coop_data):
                if coop_event:  # Assuming True means cooperation
                    coop_count += 1
                interactions_processed += 1
                if (i + 1) % interval == 0 or (i + 1) == len(coop_data):
                    current_trend.append(coop_count / interactions_processed if interactions_processed > 0 else 0)
            all_trends_for_type.append(current_trend)
            max_trend_len = max(max_trend_len, len(current_trend))

        if not max_trend_len: continue  # Skip if no data for this group type across all reps

        # Pad trends to the same length (max_trend_len) for averaging column-wise
        # replication_data_at_trend_point[trend_point_idx] = [val_repl1, val_repl2, ...]
        replication_data_at_trend_point = [[] for _ in range(max_trend_len)]
        for trend in all_trends_for_type:
            for i in range(max_trend_len):
                if i < len(trend):
                    replication_data_at_trend_point[i].append(trend[i])
                else:  # Pad with NaN if trend is shorter (or last known value)
                    replication_data_at_trend_point[i].append(trend[-1] if trend else np.nan)

        avg_trend_values, ci_lower_trend, ci_upper_trend = [], [], []
        for i in range(max_trend_len):
            mean, ci_l, ci_u = calculate_mean_ci(replication_data_at_trend_point[i])
            avg_trend_values.append(mean)
            ci_lower_trend.append(ci_l)
            ci_upper_trend.append(ci_u)

        x_trend = np.arange(len(avg_trend_values))
        if avg_trend_values:  # Check if there's anything to plot
            valid_indices_trend = [j for j, (val, l, u) in
                                   enumerate(zip(avg_trend_values, ci_lower_trend, ci_upper_trend)) if
                                   not (np.isnan(val) or np.isnan(l) or np.isnan(u))]
            if valid_indices_trend:
                ax_coop.plot([x_trend[j] for j in valid_indices_trend],
                             [avg_trend_values[j] for j in valid_indices_trend], marker='o')
                ax_coop.fill_between(
                    [x_trend[j] for j in valid_indices_trend],
                    [ci_lower_trend[j] for j in valid_indices_trend],
                    [ci_upper_trend[j] for j in valid_indices_trend],
                    alpha=0.2
                )
    plt.axhline(y=0.5, color='red', linestyle=':', linewidth=2)
    #ax_coop.set_title(f'First interaction cooperation over time (Avg over {num_replications} replications)')
    ax_coop.set_ylabel('O.S.C. rate')
    ax_coop.set_xlabel('First interactions')
    ax_coop.set_ylim(0, 1)
    ax_coop.legend()
    ax_coop.grid(True)
    plt.tight_layout()
    plt.savefig(f'first_interaction_cooperation_replications_{experiment_condition}.png')
    plt.show()


def visualize_super_additive_results_replications(final_states: List[AddableDict]):
    if not final_states:
        print("No final states provided for super_additive visualization.")
        return

    experiment_condition = final_states[0].get('experiment_condition', 'default_exp')
    num_replications = len(final_states)
    fig = plt.figure(figsize=(12, 6))

    # --- Plot 1: Intragroup vs Intergroup Cooperation Rate ---
    ax2 = fig.add_subplot(1, 1, 1)
    # Collect avg_coop_rate per match for each category from each replication
    # Then average these match averages.
    all_intragroup_match_avg_rates = []  # List of average rates from intragroup matches across all reps
    all_intergroup_match_avg_rates = []  # List of average rates from intergroup matches across all reps

    for fs in final_states:
        for match in fs.get('matches', {}).values():
            if not match.get('completed'): continue

            p1_id = str(match.get('player1_id'))
            p2_id = str(match.get('player2_id'))
            p1_info = fs.get('players', {}).get(p1_id)
            p2_info = fs.get('players', {}).get(p2_id)

            if not p1_info or not p2_info: continue
            p1_group = p1_info.get('group_id')
            p2_group = p2_info.get('group_id')

            round_results = match.get('round_results', [])
            if not round_results: continue
            num_rounds_in_match = len(round_results)

            p1_coop_count = sum(1 for move, _ in round_results if move == "action_a")
            p2_coop_count = sum(1 for _, move in round_results if move == "action_a")

            if num_rounds_in_match > 0:
                avg_coop_rate_match = ((p1_coop_count + p2_coop_count) / (2 * num_rounds_in_match)) * 100  # Percentage
            else:
                avg_coop_rate_match = np.nan  # Or 0, depending on how you want to treat no-round matches

            if not np.isnan(avg_coop_rate_match):
                if p1_group == p2_group:
                    all_intragroup_match_avg_rates.append(avg_coop_rate_match)
                else:
                    all_intergroup_match_avg_rates.append(avg_coop_rate_match)

    mean_intragroup, ci_low_intragroup, ci_up_intragroup = calculate_mean_ci(all_intragroup_match_avg_rates)
    mean_intergroup, ci_low_intergroup, ci_up_intergroup = calculate_mean_ci(all_intergroup_match_avg_rates)

    labels = ['Intragroup', 'Intergroup']
    means = [mean_intragroup if not np.isnan(mean_intragroup) else 0,
             mean_intergroup if not np.isnan(mean_intergroup) else 0]  # Use 0 for plotting if NaN

    # Calculate errors for yerr (positive values)
    lower_errors = [means[0] - (ci_low_intragroup if not np.isnan(ci_low_intragroup) else means[0]),
                    means[1] - (ci_low_intergroup if not np.isnan(ci_low_intergroup) else means[1])]
    upper_errors = [(ci_up_intragroup if not np.isnan(ci_up_intragroup) else means[0]) - means[0],
                    (ci_up_intergroup if not np.isnan(ci_up_intergroup) else means[1]) - means[1]]
    errors = [np.maximum(0, lower_errors), np.maximum(0, upper_errors)]  # Ensure errors are non-negative

    bar_positions = np.arange(len(labels))
    ax2.bar(bar_positions, means, yerr=errors, capsize=5, color=['skyblue', 'lightcoral'], tick_label=labels)
    #ax2.set_title(f'Cooperation: Intragroup vs Intergroup (Avg over {num_replications} reps of match averages)')
    ax2.set_ylabel('Cooperation rate (%)')
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(f'super_additive_results_replications_{experiment_condition}.png')
    plt.show()


def visualize_nth_results_replications(final_states: List[AddableDict]):
    if not final_states:
        print("No final states provided for Nth results visualization.")
        return

    experiment_condition = final_states[0].get('experiment_condition', 'default_exp')
    num_replications = len(final_states)
    fig = plt.figure(figsize=(15, 12))

    ax_affinity = plt.subplot(2, 1, 1)
    ax_trait = plt.subplot(2, 1, 2)
    ax_affinity.set_title(f'Mean Affinities over matches (Avg over {num_replications} replications)', fontsize=14)
    ax_trait.set_title(f'Mean Traits over matches (Avg over {num_replications} replications)', fontsize=14)

    for ax, data_key_plural, y_label_text in [
        (ax_affinity, 'SFEM', 'Affinity Value'),
        (ax_trait, 'traits', 'Trait Value')]:

        ax.set_xlabel('Matches played', fontsize=12)
        ax.set_ylabel(y_label_text, fontsize=12)

        all_metric_keys = set()
        max_matches_overall = 0
        for fs in final_states:
            for player_stats in fs.get('players', {}).values():
                metric_history = player_stats.get(data_key_plural, [])
                max_matches_overall = max(max_matches_overall, len(metric_history))
                for match_data_dict in metric_history:  # This is a list of dicts
                    if isinstance(match_data_dict, dict):
                        all_metric_keys.update(match_data_dict.keys())

        if not all_metric_keys or max_matches_overall == 0:
            ax.text(0.5, 0.5, f"No data for {data_key_plural}", horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
            continue

        colors = plt.cm.tab10(np.linspace(0, 1, len(all_metric_keys)))

        for idx, metric_type in enumerate(sorted(list(all_metric_keys))):
            # values_at_match_num_across_reps[match_idx] = [avg_val_repl1, avg_val_repl2, ...]
            values_at_match_num_across_reps = [[] for _ in range(max_matches_overall)]

            for fs in final_states:
                # player_values_for_metric_this_fs[match_idx] = [player1_val, player2_val, ...]
                player_values_for_metric_this_fs = [[] for _ in range(max_matches_overall)]
                for player_stats in fs.get('players', {}).values():
                    metric_history = player_stats.get(data_key_plural, [])
                    for match_idx, match_data_dict in enumerate(metric_history):
                        if match_idx < max_matches_overall and isinstance(match_data_dict, dict):
                            value = match_data_dict.get(metric_type)
                            if value is not None:
                                player_values_for_metric_this_fs[match_idx].append(value)

                for match_idx in range(max_matches_overall):
                    if player_values_for_metric_this_fs[match_idx]:
                        values_at_match_num_across_reps[match_idx].append(
                            np.mean(player_values_for_metric_this_fs[match_idx]))

            plot_means, plot_ci_lower, plot_ci_upper, plot_match_numbers = [], [], [], []
            for match_idx in range(max_matches_overall):
                mean, ci_l, ci_u = calculate_mean_ci(values_at_match_num_across_reps[match_idx])
                if not np.isnan(mean):  # Only add if there's valid data
                    plot_means.append(mean)
                    plot_ci_lower.append(ci_l)
                    plot_ci_upper.append(ci_u)
                    plot_match_numbers.append(match_idx + 1)  # 1-indexed matches

            if plot_means:
                color = colors[idx % len(colors)]
                ax.plot(plot_match_numbers, plot_means, marker='o', label=f"{metric_type} (Avg)", color=color,
                        linewidth=2, markersize=6)
                valid_indices_metric = [j for j, (l, u) in enumerate(zip(plot_ci_lower, plot_ci_upper)) if
                                        not (np.isnan(l) or np.isnan(u))]
                if valid_indices_metric:
                    ax.fill_between(
                        [plot_match_numbers[j] for j in valid_indices_metric],
                        [plot_ci_lower[j] for j in valid_indices_metric],
                        [plot_ci_upper[j] for j in valid_indices_metric],
                        color=color, alpha=0.2
                    )

        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if max_matches_overall > 0:
            ax.set_xticks(range(1, max_matches_overall + 1))

        # Consolidate legend to avoid duplicates from fill_between
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Will keep the line entry over the fill entry if names are identical
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10, framealpha=0.9, edgecolor='gray')

    plt.suptitle(f"Affinities and Traits", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'affinities_and_traits_replications_{experiment_condition}.png')
    plt.show()


def visualize_meta_prompt_results_replications(final_states: List[AddableDict]):
    if not final_states:
        print("No final states for meta prompt visualization.")
        return

    experiment_condition = final_states[0].get('experiment_condition', 'default_exp')
    num_replications = len(final_states)
    categories = [  # Assuming these are consistent
        'min_max', 'actions', 'payoff', 'round', 'action',
        'points', 'num_actions', 'num_points', 'tft', 'forgiving'
    ]

    # category_scores_across_reps[category_name] = [avg_score_repl1_p0, avg_score_repl2_p0, ...]
    category_scores_across_reps = {cat: [] for cat in categories}

    for fs in final_states:
        # Original code looked at player '0'. We'll maintain this.
        player_0_data = fs.get('players', {}).get('0', {})
        meta_results_p0 = player_0_data.get('meta_prompt_results')  # This is a list of dicts

        if meta_results_p0 and isinstance(meta_results_p0, list):
            for category in categories:
                scores_this_rep_cat = [d.get(category) for d in meta_results_p0 if
                                       isinstance(d, dict) and d.get(category) is not None]
                if scores_this_rep_cat:
                    category_scores_across_reps[category].append(np.mean(scores_this_rep_cat))
                # else: category_scores_across_reps[category].append(np.nan) # if want to keep track of reps with no data for a category

    plot_cats, plot_means, plot_ci_low, plot_ci_up = [], [], [], []
    for category in categories:
        mean, ci_l, ci_u = calculate_mean_ci(category_scores_across_reps[category])
        if not np.isnan(mean):  # Only plot categories with valid mean data
            plot_cats.append(category)
            plot_means.append(mean)
            plot_ci_low.append(ci_l)
            plot_ci_up.append(ci_u)

    if not plot_cats:
        print(f"No valid meta prompt data to plot for {experiment_condition}")
        return

    plt.figure(figsize=(12, 7))

    lower_errors = [m - l if not (np.isnan(m) or np.isnan(l)) else 0 for m, l in zip(plot_means, plot_ci_low)]
    upper_errors = [u - m if not (np.isnan(m) or np.isnan(u)) else 0 for m, u in zip(plot_means, plot_ci_up)]
    errors = [np.maximum(0, lower_errors), np.maximum(0, upper_errors)]

    plt.errorbar(plot_cats, plot_means, yerr=errors, fmt='o', markersize=8, capsize=5)

    plt.xlabel("Comprehension Questions")
    plt.ylabel(f"Accuracy")
    plt.xticks(rotation=90)
    plt.yticks()
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'meta_prompts_replications_{experiment_condition}.png')
    plt.show()


def visualize_data_replications(final_states_list: List[AddableDict]) -> None:
    if not final_states_list:
        print("No final_states provided to visualize_data_replications.")
        return

    exp_cond = final_states_list[0].get('experiment_condition', 'unknown_experiment')
    print(f"Starting replications visualization for experiment: {exp_cond} ({len(final_states_list)} replications)")

    visualize_tournament_results_replications(final_states_list)
    visualize_super_additive_results_replications(final_states_list)
    visualize_nth_results_replications(final_states_list)
    visualize_meta_prompt_results_replications(final_states_list)

    print("All replication visualizations generated.")


def calculate_overall_mean_action_a_rate(final_states, null_hypothesis_value=None):
    """
    Calculate the mean action A rate across all rounds and replications,
    with optional statistical significance measures. Also calculates
    first interaction cooperation rates (combined intergroup and intragroup).

    Args:
        final_states: List of AddableDict containing tournament results
        null_hypothesis_value: The value to test against. If None, no significance test is performed.
                              (default: None)

    Returns:
        dict: Contains mean action A rate, first interaction rates, and confidence intervals.
              If null_hypothesis_value is provided, also includes p-value and significance result.
    """
    import numpy as np
    from scipy import stats

    if not final_states:
        print("No data provided for analysis.")
        return None

    # List to store all action A rates across all players, all rounds, all replications
    all_action_a_rates = []

    # For first interaction cooperation analysis
    all_first_interaction_coop_events = []

    # Iterate through each replication
    for fs in final_states:
        # For overall action A rates
        for player_id, player_stats in fs.get('players', {}).items():
            # Get each player's action A rate for each round
            action_a_rates = player_stats.get('action_a_rate_history', [])
            # Convert percentages to proportion if needed (0-1 scale)
            action_a_rates = [rate / 100 if rate > 1 else rate for rate in action_a_rates]
            all_action_a_rates.extend(action_a_rates)

        # For first interaction cooperation rates (combining intergroup and intragroup)
        inter_data = fs.get('first_interaction_coop_intergroup', [])
        intra_data = fs.get('first_interaction_coop_intragroup', [])

        # Merge intergroup and intragroup data in alternating fashion, as in the original code
        coop_data = []
        i, j = 0, 0
        while i < len(inter_data):
            if j < len(intra_data):
                coop_data.extend([intra_data[j], intra_data[j + 1]])
                j += 2
            coop_data.extend([inter_data[i], inter_data[i + 1]])
            i += 2

        # Add cooperation events to overall list
        all_first_interaction_coop_events.extend([1 if event else 0 for event in coop_data])

    # Remove any NaN values for overall action A rates
    all_action_a_rates = [rate for rate in all_action_a_rates if not np.isnan(rate)]

    # Process results for overall action A rates
    result = {}

    if all_action_a_rates:
        # Calculate overall mean
        mean_rate = np.mean(all_action_a_rates)

        # Calculate 95% confidence interval
        n_rates = len(all_action_a_rates)
        sem_rates = stats.sem(all_action_a_rates)
        confidence = 0.95
        ci_lower_rate, ci_upper_rate = stats.t.interval(confidence, n_rates - 1, loc=mean_rate, scale=sem_rates)

        # Store results for overall rates
        result['mean_action_a_rate'] = mean_rate
        result['action_a_confidence_interval'] = (ci_lower_rate, ci_upper_rate)
        result['action_a_sample_size'] = n_rates

        # Print basic results for overall rates
        print("=== Overall Action A Rate ===")
        print(f"RI & {mean_rate:.2f} & [{ci_lower_rate:.2f}, {ci_upper_rate:.2f}] \\\\")

        # Perform t-test against null hypothesis only if a value is provided
        if null_hypothesis_value is not None:
            t_stat, p_value = stats.ttest_1samp(all_action_a_rates, null_hypothesis_value)
            result['action_a_p_value'] = p_value
            result['action_a_significant_diff_from_null'] = p_value < 0.05
            result['null_hypothesis_value'] = null_hypothesis_value

            # Print significance test results
            print(f"P-value (vs. {null_hypothesis_value * 100:.0f}%): {p_value:.6f}")
            print(
                f"Statistically {'significant' if p_value < 0.05 else 'not significant'} difference from {null_hypothesis_value * 100:.0f}%")
    else:
        print("No valid action A rates found.")

    # Process first interaction cooperation data
    if all_first_interaction_coop_events:
        # Calculate mean first interaction cooperation rate
        mean_first_coop = np.mean(all_first_interaction_coop_events)

        # Calculate 95% confidence interval
        n_first = len(all_first_interaction_coop_events)
        sem_first = stats.sem(all_first_interaction_coop_events)
        ci_lower_first, ci_upper_first = stats.t.interval(confidence, n_first - 1, loc=mean_first_coop, scale=sem_first)

        # Store results for first interaction rates (already in 0-1 scale)
        result['mean_first_interaction_coop_rate'] = mean_first_coop  # Convert to percentage
        result['first_interaction_confidence_interval'] = (
        ci_lower_first, ci_upper_first)  # Convert to percentage
        result['first_interaction_sample_size'] = n_first

        # Print results for first interaction cooperation
        print("\n=== First Interaction Cooperation Rate (Combined) ===")
        print(f"RI & {mean_first_coop:.2f} & [{ci_lower_first:.2f}, {ci_upper_first:.2f}] \\\\")

        # Perform t-test against null hypothesis if provided
        if null_hypothesis_value is not None:
            t_stat_first, p_value_first = stats.ttest_1samp(all_first_interaction_coop_events, null_hypothesis_value)
            result['first_interaction_p_value'] = p_value_first
            result['first_interaction_significant_diff_from_null'] = p_value_first < 0.05

            # Print significance test results
            print(f"P-value (vs. {null_hypothesis_value * 100:.0f}%): {p_value_first:.6f}")
            print(
                f"Statistically {'significant' if p_value_first < 0.05 else 'not significant'} difference from {null_hypothesis_value * 100:.0f}%")
    else:
        print("No valid first interaction cooperation data found.")

    return result
