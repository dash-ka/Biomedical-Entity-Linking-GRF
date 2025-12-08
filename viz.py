import pandas as pd
import matplotlib.pyplot as plt

def plot_recall_from_long_format(csv_path, target_k_synonyms=10, target_alpha=0.5, cutoffs=[1, 2, 3, 5, 10]):
    # Load main data
    df = pd.read_csv(csv_path)
    df_syn = df[(df["strategy"].isin(["RF-GRF_synonyms", "VF-GRF_synonyms"]))  & 
                (df['depth'] == target_k_synonyms) &
                (df['alpha'] == target_alpha) & 
                (df["metric"] == "recall@cutoff") &
                (df["cutoff"].isin(cutoffs))]

    # Load definitions data
   
    df_def = df[(df["strategy"].isin(["RF-GRF_definition", "VF-GRF_definition"]))  & 
                (df['alpha'] == target_alpha) &
                (df["metric"] == "recall@cutoff") &
                (df["cutoff"].isin(cutoffs))]

    # Load standard/baseline data
    
    df_std_name = df[(df["strategy"].isin(["RF-GRF_standard_name", "VF-GRF_standard_name"]))  & 
                    (df['alpha'] == target_alpha) &
                    (df["metric"] == "recall@cutoff") &
                    (df["cutoff"].isin(cutoffs))]

    
    df_all = df[
            (df["strategy"].isin(["RF-GRF_synonyms+definition+standard_name", "VF-GRF_synonyms+definition+standard_name"]))  & 
            (df['alpha'].isin([target_alpha, -1])) &
            (df['depth'] == target_k_synonyms) &
            (df["metric"] == "recall@cutoff") &
            (df["cutoff"].isin(cutoffs))  
        ] 
        
    #Baseline (k_synonyms=5, alpha=1, GRF)
    baseline = df[
         (df['strategy'] == "VF-GRF_standard_name") &
         (df['alpha'] == 1) &
         (df["metric"] == "recall@cutoff") &
         (df["cutoff"].isin(cutoffs))
         ]
        
       
    # Assign colors based on file origin
    file_color_map = {
        'main':"#1B756F",#"#87C842",   
        'standard':"#9358F2",    
        'definitions': "#FFAE4A",
        'all':"#44CBF9"
    }

    # Line styles per strategy
    style_map = {
        'VF': '-',
        'RF': '--',
    }

    plt.figure(figsize=(4, 6))

    # Plot main CSV
    for strategy in df_syn['strategy'].unique():
        sub_df = df_syn[df_syn['strategy'] == strategy]
        parts = strategy.split("-")
        label_strategy = "Vector GRF" if parts[0] == "VF" else "Rank GRF"
        label = f"{label_strategy} (10 synonyms)"
        plt.plot(sub_df['cutoff'], sub_df['mean'],
                marker='o', markersize=7,
                    markeredgecolor='white',   #  white border around markers
                    markeredgewidth=0.5,       #  thickness of white border
                label=label,
                color=file_color_map['main'],
                linestyle=style_map.get(parts[0], '--'))

    # Plot definitions CSV
    for strategy in df_all['strategy'].unique():
        sub_df = df_all[df_all['strategy'] == strategy]
        parts = strategy.split("-")
        label_strategy = "Vector GRF" if parts[0] == "VF" else "Rank GRF"
        label = f"{label_strategy} (All-in-One)"
        plt.plot(sub_df['cutoff'], sub_df['mean'],
                    marker='o', markersize=7,
                    markeredgecolor='white',   #  white border around markers
                    markeredgewidth=0.5,       #  thickness of white border
                    label=label,
                    color=file_color_map['all'],
                    linestyle=style_map.get(parts[0], '--'))

    # Plot definitions CSV
    for strategy in df_def['strategy'].unique():
        
        sub_df = df_def[df_def['strategy'] == strategy]
        parts = strategy.split("-")
        label_strategy = "Vector GRF" if parts[0] == "VF" else "Rank GRF"
        label = f"{label_strategy} (definition)"
        plt.plot(sub_df['cutoff'], sub_df['mean'],
                    marker='o', markersize=7,
                    markeredgecolor='white',   #  white border around markers
                    markeredgewidth=0.5,       #  thickness of white border
                    label=label,
                    color=file_color_map['definitions'],
                    linestyle=style_map.get(parts[0], '--'))
            

    for strategy in df_std_name['strategy'].unique():
        sub_df = df_std_name[df_std_name['strategy'] == strategy]
        parts = strategy.split("-")
        
        label_strategy = "Vector GRF" if parts[0] == "VF" else "Rank GRF"
        label = f"{label_strategy} (standard name)"
        plt.plot(sub_df['cutoff'], sub_df['mean'],
                marker='o', markersize=7,
                    markeredgecolor='white',   #  white border around markers
                    markeredgewidth=0.5,       #  thickness of white border
                label=label,
                color=file_color_map['standard'],
                linestyle=style_map.get(parts[0], '--'))

    
        
    baseline_values = baseline["mean"].values
    plt.plot(cutoffs, baseline_values,  marker='o', markersize=6, color="#A4ACAD", linewidth=1, label='Baseline')

    # Final touches
    plt.xticks(cutoffs)
    plt.xlabel("Cutoff")
    plt.ylim(0.59, 1)
    plt.legend(title="Strategy (Feedback Type)",loc="lower right", fontsize=12, title_fontsize=14)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.show()

    
def plot_recall_vs_alpha(csv_path): 
    
    # Setup figure with 3 subplots
    recall_ks = [1, 3, 5]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    # Define styles
    linestyle_map = {
        "Vector": "-",
        "Rank": "--"
    }

    color_map = {
        1: "steelblue",
        3: "#90DAE6",
        5: "#62A4AD",
        10: "#1B756F",
        "def": "#FFAE4A",
        "name": "#9358F2",
    }

    # Plot each Recall@k
    df = pd.read_csv(csv_path)
    for ax, recall_k in zip(axes, recall_ks):
        filtered_df = df[(df['cutoff'] == recall_k) & (df["metric"] == "recall@cutoff")]

        df_syn = filtered_df[(filtered_df["strategy"].isin(["RF-GRF_synonyms", "VF-GRF_synonyms"]))  & 
                    (filtered_df['depth'].isin([3, 5, 10])) &
                    (filtered_df["metric"] == "recall@cutoff") &
                    (filtered_df["cutoff"] == recall_k)]

        # Load definitions data
        df_def = filtered_df[(filtered_df["strategy"].isin(["RF-GRF_definition", "VF-GRF_definition"]))  & 
                    (filtered_df["metric"] == "recall@cutoff") &
                    (filtered_df["cutoff"] == recall_k)]

        # Load standard/baseline data        
        df_std_name = filtered_df[(filtered_df["strategy"].isin(["RF-GRF_standard_name", "VF-GRF_standard_name"]))  & 
                    (filtered_df["metric"] == "recall@cutoff") &
                    (filtered_df["cutoff"] == recall_k)]

        for strategy in df_syn["strategy"].unique():#filtered_df['strategy'].unique():# ["PRF (VF)"]:#
            
            for k in sorted(df_syn['depth'].unique())[:4]:
                subset = df_syn[
                        (df_syn['strategy'] == strategy) &
                        (df_syn['depth'] == k)
                    ].sort_values(by="alpha")
          
                if not subset.empty:
                    strategy_label = "Vector" if "VF" in strategy else "Rank"

                    label = f"{strategy_label} ({int(k)} synonyms)"
                    ax.plot(
                        subset['alpha'],
                        subset['mean'],
                        linestyle=linestyle_map.get(strategy_label, "-"),
                        color=color_map.get(k, 'gray'),
                        marker='o',
                        label=label
                    )
            
        for strategy in df_def["strategy"].unique():#filtered_df['strategy'].unique():# ["PRF (VF)"]:#
            
            subset = df_def[(df_def['strategy'] == strategy)].sort_values(by="alpha")
            if not subset.empty:
                strategy_label = "Vector" if "VF" in strategy else "Rank"

                label = f"{strategy_label} (definition)"
                ax.plot(
                        subset['alpha'],
                        subset['mean'],
                        linestyle=linestyle_map.get(strategy_label, "-"),
                        color=color_map.get("def", 'gray'),
                        marker='o',
                        label=label
                    )
        for strategy in df_std_name["strategy"].unique():#filtered_df['strategy'].unique():# ["PRF (VF)"]:#
            
            subset = df_std_name[(df_std_name['strategy'] == strategy)].sort_values(by="alpha")
            if not subset.empty:
                strategy_label = "Vector" if "VF" in strategy else "Rank"

                label = f"{strategy_label} (standard_name)"
                ax.plot(
                        subset['alpha'],
                        subset['mean'],
                        linestyle=linestyle_map.get(strategy_label, "-"),
                        color=color_map.get("name", 'gray'),
                        marker='o',
                        label=label
                    )
                
        ax.set_xlabel("Alpha")
        ax.set_xticks([round(a, 1) for a in list(pd.Series([i * 0.1 for i in range(11)]))])
        ax.grid(True)

    # Y-axis label only for the first subplot
    axes[0].set_ylabel("Recall")
    axes[2].legend(title="Strategy (Feedback Type)", loc='lower right', fontsize = 12, title_fontsize = 14)

    plt.suptitle("Recall@k vs Alpha", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



def plot_spider_plots(feedback_type, system_name):
    """
    Generate radar (spider) plots of retrieval accuracy for a selected feedback type
    and LLM family.

    This function visualizes the Recall@1 performance across multiple datasets using
    GRF feedback generated by different model sizes within a given LLM family.
    Results for the chosen open-source models are compared against GPT-4o baselines.

    Args:
        feedback_type (str):
            The type of generative feedback to visualize. Must be one of:
            ["synonyms", "definition", "standard_name", "aio"].
            - "synonyms": Uses synonym-based feedback
            - "definition": Uses definition-based feedback
            - "standard_name": Uses standardized term generation
            - "aio": Uses the All-in-One vector-based GRF strategy

        system_name (str):
            The LLM family to compare. Must be one of:
            ["gemma3", "qwen3"].
            - "gemma3": Compares Gemma-3 (12B, 4B, 1B) models to GPT-4o
            - "qwen3": Compares Qwen-3 (14B, 4B, 1.7B) models to GPT-4o

    Returns:
        None. Displays a matplotlib radar plot.
    """

    if system_name == "gemma3":
        models = [ "GPT-4o", "Gemma3-12b", "Gemma3-4b", "Gemma3-1b"]
        colors = {
            "Gemma3-12b": "#414CB1",   # red
            "Gemma3-4b": "#9358F2",    # blue
            "Gemma3-1b": "#44CBF9",  # green
            "GPT-4o": "#F99421"        # purple (dashed)
        }
        
        linestyles = {
            "Gemma3-12b": "-",
            "Gemma3-4b": "-",
            "Gemma3-1b": "-",
            "GPT-4o": "--"    # dashed
        }
        if feedback_type == "synonyms":
            data = {
            "model": ["GPT-4o"]* 8 + ["Gemma3-12b"]*8 + ["Gemma3-4b"]*8 + ["Gemma3-1b"]*8,
            "dataset": ["NCBI-Disease", "BC5CDR\nDisease", "BC5CDR\nChemical", "NLM-Chem", 
                        "NLM-Gene", "GNormPlus", "S800", "Linnaeus"] * 4,
            "accuracy": [
                0.7378, 0.7851, 0.8907, 0.7637, 0.7455, 0.3608, 0.6830, 0.7889,
                0.7359, 0.7780, 0.8946, 0.7653, 0.7430, 0.3514, 0.6669, 0.7535,
                0.7320, 0.7743, 0.8736, 0.7530, 0.7275, 0.3458, 0.6535, 0.7546,
                0.7407, 0.7496, 0.8929, 0.7236, 0.6925, 0.3557, 0.7016, 0.6584
        
                ]
        }
        elif feedback_type == "definition":
            data = {
            "model": ["GPT-4o"]* 8 + ["Gemma3-12b"]*8 + ["Gemma3-4b"]*8 + ["Gemma3-1b"]*8,
            "dataset": ["NCBI-Disease", "BC5CDR\nDisease", "BC5CDR\nChemical", "NLM-Chem", 
                        "NLM-Gene", "GNormPlus", "S800", "Linnaeus"] * 4,
            "accuracy": [
                0.7135, 0.7700, 0.8715, 0.7497, 0.7347, 0.3818, 0.7042, 0.7624,
                0.7233, 0.7654, 0.8565, 0.7361, 0.7328, 0.3661, 0.6866, 0.7679,
                0.7038, 0.7777, 0.8629, 0.7351, 0.7288, 0.3834, 0.6760, 0.7513,
                0.7330, 0.7635, 0.8993, 0.7265, 0.7102, 0.3615, 0.7127, 0.6795
        
                ]
        }
        elif feedback_type == "standard_name":
            data = {
            "model": ["GPT-4o"]* 8 + ["Gemma3-12b"]*8 + ["Gemma3-4b"]*8 + ["Gemma3-1b"]*8,
            "dataset": ["NCBI-Disease", "BC5CDR\nDisease", "BC5CDR\nChemical", "NLM-Chem", 
                        "NLM-Gene", "GNormPlus", "S800", "Linnaeus"] * 4,
            "accuracy": [
                0.6796, 0.7453, 0.8758, 0.7226, 0.7416, 0.3865, 0.7570, 0.7790,
                0.6941, 0.7546, 0.8586, 0.7351, 0.7151, 0.3427, 0.7147, 0.7127,
                0.6796, 0.7577, 0.8565, 0.7028, 0.6876, 0.3286, 0.6936, 0.6519,
                0.6893, 0.7330, 0.8479, 0.7194, 0.6984, 0.3302, 0.6408, 0.5669
        
                ]
        }
        elif feedback_type =="aio":
            data = {
            "model": ["GPT-4o"]* 8 + ["Gemma3-12b"]*8 + ["Gemma3-4b"]*8 + ["Gemma3-1b"]*8,
            "dataset": ["NCBI-Disease", "BC5CDR\nDisease", "BC5CDR\nChemical", "NLM-Chem", 
                        "NLM-Gene", "GNormPlus", "S800", "Linnaeus"] * 4,
            "accuracy": [
                0.7281, 0.7657, 0.9387, 0.7753, 0.7876, 0.3928, 0.7711, 0.8430,
                0.7165, 0.7796, 0.9169, 0.7655, 0.7499, 0.3809, 0.7387, 0.7966,
                0.7203, 0.7648, 0.9066, 0.7537, 0.7367, 0.3776, 0.7147, 0.7679,
                0.7330, 0.7635, 0.8993, 0.7265, 0.7102, 0.3615, 0.7127, 0.6795
        
                ]
        }
        else:
            raise ValueError(f"Incorrect feedback type: {feedback_type}")
            
    elif: system_name == "qwen3":
        models = [ "GPT-4o", "Qwen3-14b", "Qwen3-4b", "Qwen3-1.7b"]
        colors = {
            "Qwen3-14b": "#414CB1",   # red
            "Qwen3-4b": "#9358F2",    # blue
            "Qwen3-1.7b": "#44CBF9",  # green
            "GPT-4o": "#F99421"        # purple (dashed)
        }
        
        linestyles = {
            "Qwen3-14b": "-",
            "Qwen3-4b": "-",
            "Qwen3-1.7b": "-",
            "GPT-4o": "--"    # dashed
        }
        if feedback_type == "synonyms":
            data = {
            "model": ["GPT-4o"]* 8 + ["Qwen3-14b"]*8 + ["Qwen3-4b"]*8 + ["Qwen3-1.7b"]*8,
            "dataset": ["NCBI-Disease", "BC5CDR\nDisease", "BC5CDR\nChemical", "NLM-Chem", 
                        "NLM-Gene", "GNormPlus", "S800", "Linnaeus"] * 4,
            "accuracy": [
                0.7378, 0.7851, 0.8907, 0.7637, 0.7455, 0.3608, 0.6830, 0.7889,
                0.7271, 0.7768, 0.8972, 0.7591, 0.7369, 0.3624, 0.6830, 0.8143,
                0.7155, 0.7564, 0.9032, 0.7534, 0.7341, 0.3531, 0.6718, 0.7657,
                0.6854, 0.7166, 0.8492, 0.7069, 0.7128, 0.3578, 0.6401, 0.7403,
               
            ]
        }
        elif feedback_type == "definition":
            data = {
            "model": ["GPT-4o"]* 8 + ["Qwen3-14b"]*8 + ["Qwen3-4b"]*8 + ["Qwen3-1.7b"]*8,
            "dataset": ["NCBI-Disease", "BC5CDR\nDisease", "BC5CDR\nChemical", "NLM-Chem", 
                        "NLM-Gene", "GNormPlus", "S800", "Linnaeus"] * 4,
            "accuracy": [
                0.7135, 0.7700, 0.8715, 0.7497, 0.7347, 0.3818, 0.7042, 0.7624,
                0.7087, 0.7669, 0.8522, 0.7330, 0.7377, 0.3896, 0.6760, 0.7513,
                0.7233, 0.7669, 0.8672, 0.7372, 0.7249, 0.3912, 0.6830, 0.7513,
                0.7184, 0.7700, 0.8458, 0.7247, 0.7210, 0.3787, 0.6901, 0.7237,
               
            ]
        }
        elif feedback_type == "standard_name":
            data = {
            "model": ["GPT-4o"]* 8 + ["Qwen3-14b"]*8 + ["Qwen3-4b"]*8 + ["Qwen3-1.7b"]*8,
            "dataset": ["NCBI-Disease", "BC5CDR\nDisease", "BC5CDR\nChemical", "NLM-Chem", 
                        "NLM-Gene", "GNormPlus", "S800", "Linnaeus"] * 4,
            "accuracy": [
                0.6796, 0.7453, 0.8758, 0.7226, 0.7416, 0.3865, 0.7570, 0.7790,
                0.7135, 0.7777, 0.9079, 0.7612, 0.7102, 0.3474, 0.7359, 0.7127,
                0.7281, 0.7623, 0.9079, 0.7601, 0.7072, 0.3427, 0.6971, 0.7127,
                0.7233, 0.7777, 0.8586, 0.7080, 0.7023, 0.3333, 0.6514, 0.6132,
               
            ]
        }
        elif feedback_type =="aio":
            data = {
                "model": ["GPT-4o"]* 8 + ["Qwen3-14b"]*8 + ["Qwen3-4b"]*8 + ["Qwen3-1.7b"]*8,
                "dataset": ["NCBI-Disease", "BC5CDR\nDisease", "BC5CDR\nChemical", "NLM-Chem", 
                            "NLM-Gene", "GNormPlus", "S800", "Linnaeus"] * 4,
                "accuracy": [
                    0.7281, 0.7657, 0.9387, 0.7753, 0.7876, 0.3928, 0.7711, 0.8430,
                    0.7330, 0.7916, 0.9280, 0.7760, 0.7603, 0.3802, 0.7549, 0.8066,
                    0.7262, 0.7768, 0.9263, 0.7714, 0.7465, 0.3802, 0.7218, 0.7790,
                    0.7194, 0.7645, 0.8843, 0.7424, 0.7390, 0.3611, 0.6887, 0.7491
                ]
            }
        else:
            raise ValueError(f"Incorrect feedback type: {feedback_type}")
            
    else:
        raise ValueError(f"Incorrect model name: {system_name}")

    df = pd.DataFrame(data)
    
    # pivot
    pivot_df = df.pivot_table(index="model", columns="dataset", values="accuracy", aggfunc="mean")
    
    #datasets = sorted(pivot_df.columns)
    datasets = ['BC5CDR\nChemical',
     'BC5CDR\nDisease',
     'GNormPlus',
     'Linnaeus',
     'NLM-Chem',
     'NCBI-Disease',
     'NLM-Gene',
     'S800']
    
    # angles for radar
    angles = np.linspace(0, 2*np.pi, len(datasets), endpoint=False).tolist()
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # pad angular labels so they don’t overlap datapoints
    ax.tick_params(axis='x', pad=10)
    
    # Optional: rotate labels slightly for readability
    for label in ax.get_xticklabels():
        label.set_rotation(20)
    
    for model in models:
        values = pivot_df.loc[model, datasets].tolist()
        values += values[:1]
        ax.plot(
        angles, values,
        marker="o",
        markersize=8,
        markerfacecolor=colors.get(model, "#000000"),
        markeredgecolor="white",     # white border
        markeredgewidth=0.8,         # thickness of border
        color=colors.get(model, "#000000"),
        linestyle=linestyles.get(model, "-"),
        linewidth=1,
        label=model
    )
    
    # Make grid lines and circular axes light gray
    ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.xaxis.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.spines['polar'].set_color('gray')
    ax.spines['polar'].set_alpha(0.5)
    
    # formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets)
    
    # Set radial ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(0, 1)
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
    #ax.set_title("Model Accuracy Across Datasets (Radar Plot)", size=14)
    #ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.show()
