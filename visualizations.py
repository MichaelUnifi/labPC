import json
import matplotlib.pyplot as plt
import numpy as np

def load_json(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Errore caricamento {filename}: {e}")
        return None

def save_sequential_latex_table(sequential_data, output_file='sequential_perf_table.tex'):
    if not sequential_data: return

    with open(output_file, 'w') as f:
        f.write(r"\begin{table*}[ht]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\begin{tabular}{|l|r|r|r|c|r|}" + "\n")
        f.write(r"\hline" + "\n") # Linea superiore
        f.write(r"\textbf{Scenario} & \textbf{Doc Size} & \textbf{Band} & \textbf{Tables} & \textbf{Ins. Time (s)} & \textbf{Qry. Time (s)} \\ \hline" + "\n")
        f.write(r"\hline" + "\n") # Doppia linea sotto l'header (opzionale)

        for res in sequential_data['results']:
            scenario = res['name'].split('_')[0].capitalize()
            d_size = res.get('doc_size', '-')
            b_size = res.get('band_size', '-')
            t_size = res.get('num_tables', '-')
            
            # 2. Aggiunta di \hline dopo ogni riga di dati
            f.write(f"{scenario} & {d_size} & {b_size} & {t_size} & {res['avg_insertion_time']:.4f} & {res['avg_query_time']:.4f} \\\\ \\hline\n")

        f.write(r"\end{tabular}" + "\n")
        f.write(r"\caption{Sequential benchmark results on the various LSH scenarios.}" + "\n")
        f.write(r"\label{tab:lsh_seq_baseline}" + "\n")
        f.write(r"\end{table*}" + "\n")

def plot_speedup_analysis(parallel_data, sequential_data):
    if not parallel_data or not sequential_data: return

    # --- PARAMETRI DI ALTA RISOLUZIONE ---
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.antialiased'] = True  # Massima nitidezza tratti
    # -------------------------------------

    seq_map = {r['name']: (r['avg_insertion_time'], r['avg_query_time']) 
               for r in sequential_data['results']}
    
    groups = {
        'size':  ('Speedup varying doc size', 'doc size'),
        'tables': ('Speedup varying number of tables', 'tables'),
        'band':   ('Speedup varying band size', 'band size')
    }

    colors = ['#E64B35FF', '#4DBBD5FF', '#00A087FF', '#3C5488FF', '#F39B7FFF'] # Palette scientifica (Nature/Lancet)

    for prefix, (title, label_unit) in groups.items():
        configs = [r for r in parallel_data['results'] if r['name'].startswith(prefix)]
        if not configs: continue

        # Dimensioni generose per evitare compressione dei pixel
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor('white') # Sfondo pulito
        fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)

        all_threads = set()

        for idx, cfg in enumerate(configs):
            name = cfg['name']
            if name not in seq_map: continue
            
            base_ins, base_qry = seq_map[name]
            threads = [tr['threads'] for tr in cfg['thread_results']]
            all_threads.update(threads)
            
            s_ins = [base_ins / tr['avg_insertion_time'] for tr in cfg['thread_results']]
            s_qry = [base_qry / tr['avg_query_time'] for tr in cfg['thread_results']]
            
            val = name.split('_')[-1]
            label = f"{label_unit.title()} {val}"
            color = colors[idx % len(colors)]
            
            # Marker con 'markeredgecolor' bianco crea un distacco netto (effetto alta risoluzione)
            ax1.plot(threads, s_ins, 'o-', linewidth=3, markersize=10, 
                    label=label, color=color, markeredgecolor='white', markeredgewidth=1.5)
            ax2.plot(threads, s_qry, 's-', linewidth=3, markersize=10, 
                    label=label, color=color, markeredgecolor='white', markeredgewidth=1.5)

        sorted_threads = sorted(list(all_threads))

        for ax, subtitle in zip([ax1, ax2], ['Insertion', 'Query']):
            # Linea ideale scura per contrasto
            ax.plot(sorted_threads, sorted_threads, color='#333333', linestyle='--', 
                    linewidth=1.5, alpha=0.7, label='Ideal')
            
            ax.set_title(f'{subtitle} Speedup', fontsize=15, pad=15)
            ax.set_xlabel('Threads', fontweight='600')
            ax.set_ylabel('Speedup Factor', fontweight='600')
            ax.set_xticks(sorted_threads)
            
            # Griglia sottile ma definita
            ax.grid(True, linestyle=(0, (5, 10)), color='lightgray')
            ax.legend(frameon=True, facecolor='white', framealpha=1)
            
            # Pulizia bordi
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        plt.tight_layout()
        filename = f'speedup_{prefix}.png'
        
        # Salvataggio con 'bbox_inches' per evitare pixel bianchi inutili
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=False)
        plt.close()

    print("Immagini HQ generate.")

if __name__ == "__main__":
    # Caricamento dati
    s_data = load_json('results_sequential_scaling.json')
    p_data = load_json('results_parallel_scaling.json')
    
    # 1. Tabella LaTeX (Solo Sequenziale)
    save_sequential_latex_table(s_data)
    
    # 2. Analisi Speedup (Confronto Parallelo vs Sequenziale)
    plot_speedup_analysis(p_data, s_data)

    print("\nBenchmark visualization completed.")