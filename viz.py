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
                    markeredgecolor='white',   # ✅ white border around markers
                    markeredgewidth=0.5,       # ✅ thickness of white border
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
                    markeredgecolor='white',   # ✅ white border around markers
                    markeredgewidth=0.5,       # ✅ thickness of white border
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
                    markeredgecolor='white',   # ✅ white border around markers
                    markeredgewidth=0.5,       # ✅ thickness of white border
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
                    markeredgecolor='white',   # ✅ white border around markers
                    markeredgewidth=0.5,       # ✅ thickness of white border
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
                
        #ax.set_title(f"Recall@{recall_k} vs Alpha")
        ax.set_xlabel("Alpha")
        ax.set_xticks([round(a, 1) for a in list(pd.Series([i * 0.1 for i in range(11)]))])
        ax.grid(True)

    # Y-axis label only for the first subplot
    axes[0].set_ylabel("Recall")
    axes[2].legend(title="Strategy (Feedback Type)", loc='lower right', fontsize = 12, title_fontsize = 14)

    plt.suptitle("Recall@k vs Alpha", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
