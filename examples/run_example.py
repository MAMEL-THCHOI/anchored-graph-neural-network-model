"""Calculate anchors, rebuild graphs, train and evaluate the bundled example."""
from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples'
sys.path.insert(0, str(ROOT))
from anchored_gnn.graph import build_grain_graph
from anchored_gnn.anchors import references_from_microstructure


def run(module, arguments, env):
    command = [sys.executable, '-E', '-u', '-m', module, *map(str, arguments)]
    print('Running: ' + ' '.join([module, *map(str, arguments)]), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    config = json.loads((ROOT / 'configs/training.json').read_text())
    work = config['work_levels']
    angles = np.arange(0, 180, 20)
    with np.load(EXAMPLE / 'targets.npz', allow_pickle=False) as targets, np.load(output / 'predictions.npz', allow_pickle=False) as predictions:
        ids = predictions['condition_ids'].tolist()
        indices = [targets['condition_ids'].tolist().index(i) for i in ids]
        true_h, pred_h = targets['hardening_radius'][indices], predictions['hardening_radius']
        true_r, pred_r = targets['r_values'][indices], predictions['r_values']
        hardening_rows, r_rows = [], []
        for n, condition in enumerate(ids):
            for w, wp in enumerate(work):
                for a, angle in enumerate(angles):
                    hardening_rows.append([condition, wp, angle, true_h[n,w,a], pred_h[n,w,a], pred_h[n,w,a]-true_h[n,w,a]])
            for a, angle in enumerate([0,45,90]):
                r_rows.append([condition, angle, true_r[n,a], pred_r[n,a], pred_r[n,a]-true_r[n,a]])
        write_csv(output/'test_hardening_comparison.csv', ['condition_id','work_mj_m3','loading_angle_deg','cpfem_mpa','predicted_mpa','difference_mpa'], hardening_rows)
        write_csv(output/'test_r_comparison.csv', ['condition_id','tensile_angle_deg','r_cpfem','r_predicted','difference'], r_rows)
    history = json.loads((output/'history.json').read_text())
    write_csv(output/'training_history.csv', list(history[0]), [[row[k] for k in history[0]] for row in history])
    metrics = json.loads((output/'test_metrics.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1, 3, figsize=(12,3.8), layout='constrained')
    for n, condition in enumerate(ids):
        axes[0].scatter(true_h[n], pred_h[n], s=14, label=condition, color=['#222222','#c43c39','#888888'][n % 3])
    low, high = min(true_h.min(),pred_h.min()), max(true_h.max(),pred_h.max())
    axes[0].plot([low,high],[low,high], color='#888888', linestyle='--', linewidth=1)
    axes[0].set(xlabel='CPFEM stress radius (MPa)', ylabel='Predicted stress radius (MPa)', title=f"Hardening | test R² = {metrics['hardening']['R2']:.4f}")
    axes[0].legend(frameon=False)
    locations=np.arange(true_r.size)
    axes[1].plot(locations,true_r.ravel(),'o',color='#222222',label='CPFEM')
    axes[1].plot(locations,pred_r.ravel(),'x',color='#c43c39',label='Prediction',markersize=7)
    axes[1].set_xticks(locations,[f'{i}\n{angle}°' for i in ids for angle in [0,45,90]],fontsize=8)
    axes[1].set(ylabel='Lankford coefficient',title=f"Lankford | test R² = {metrics['r']['R2']:.4f}")
    axes[1].legend(frameon=False)
    axes[2].plot([r['epoch'] for r in history],[r['loss'] for r in history],color='#222222')
    axes[2].set(xlabel='Epoch',ylabel='Total training loss',title='Training on fourteen textures')
    fig.savefig(output/'example_results.png',dpi=250)
    plt.close(fig)
    table='\n'.join(f'| {i} | {angle} | {actual:.6f} | {pred:.6f} | {diff:+.6f} |' for i,angle,actual,pred,diff in r_rows)
    summary=json.loads((output/'summary.json').read_text())
    (output/'RESULTS.md').write_text(
        '# Bundled example results\n\n'
        'Fresh training on fourteen textures; three validation textures select the checkpoint; three separate test textures are evaluated afterward. '
        'This small example demonstrates the software workflow and does not reproduce the full-database paper metrics.\n\n'
        f"Seed: {summary['seed']}. Completed epochs: {summary['epochs_completed']}. Selected epoch: {summary['selected_epoch']} ({summary['selected_state_type']}).\n\n"
        '| Response | Test R² | Mean absolute error |\n|---|---:|---:|\n'
        f"| Hardening | {metrics['hardening']['R2']:.6f} | {metrics['hardening']['mae']:.6f} MPa |\n"
        f"| Lankford coefficients | {metrics['r']['R2']:.6f} | {metrics['r']['mae']:.6f} |\n\n"
        'R² is pooled over the three test RVEs and their supplied response coordinates. This sample is too small to establish generalization performance.\n\n'
        '| Condition | Tensile angle (°) | CPFEM r | Predicted r | Difference |\n|---|---:|---:|---:|---:|\n'+table+'\n\n'
        '![Example results](example_results.png)\n', encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=7,help='Random seed (default: 7; bundled reference: 7)')
    parser.add_argument('--output',type=Path,default=ROOT/'example_run')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory to preserve existing results.')
    output.mkdir(parents=True)
    graph_dir=output/'graphs'
    graph_dir.mkdir()
    selection=json.loads((EXAMPLE/'selection.json').read_text())
    systems=np.loadtxt(EXAMPLE/'slip_systems.txt')
    anchors, diagnostics = {}, {}
    for row in selection['conditions']:
        condition=row['condition_id']
        print(f'Building graph and calculating Sachs/Taylor references: {condition}', flush=True)
        with np.load(EXAMPLE/'microstructures'/f'{condition}.npz',allow_pickle=False) as raw:
            graph=build_grain_graph(raw['feature_ids'],raw['grain_eulers'],systems)
            sachs, taylor, info = references_from_microstructure(raw['feature_ids'],raw['grain_eulers'],systems)
        anchors[condition] = (sachs, taylor)
        diagnostics[condition] = info
        with np.load(EXAMPLE/'graphs'/f'{condition}.npz',allow_pickle=False) as reference:
            for key in graph: np.testing.assert_allclose(graph[key],reference[key],rtol=2e-6,atol=2e-6)
        np.savez_compressed(graph_dir/f'{condition}.npz',**graph)
    with np.load(EXAMPLE/'targets.npz', allow_pickle=False) as reference:
        targets = {key: reference[key] for key in reference.files}
    ids = targets['condition_ids']
    calculated = {'condition_ids': ids}
    max_difference = {}
    for index, key in enumerate(['anchor_lower', 'anchor_upper']):
        values = np.asarray([anchors[condition][index] for condition in ids])
        # Stored anchors are regression checks only; newly calculated values feed the model.
        np.testing.assert_allclose(values, targets[key], rtol=2e-6, atol=2e-8)
        max_difference[key] = float(np.max(np.abs(values-targets[key])))
        targets[key] = values
        calculated[key] = values
    np.savez_compressed(output/'anchors.npz', **calculated)
    np.savez_compressed(output/'targets.npz', **targets)
    with (EXAMPLE/'split.csv').open(newline='', encoding='utf-8') as stream:
        test_ids = [row['condition_id'] for row in csv.DictReader(stream) if row['split'] == 'test']
    test_indices = [ids.tolist().index(condition) for condition in test_ids]
    np.savez_compressed(output/'test_anchors.npz', **{key: value[test_indices] for key,value in calculated.items()})
    (output/'anchor_diagnostics.json').write_text(json.dumps({
        'max_absolute_difference_from_stored_references': max_difference,
        'taylor': diagnostics}, indent=2), encoding='utf-8')
    print('Rebuilt and checked all 20 graphs and anchor sets; using calculated anchors.',flush=True)
    # Numerical runtime settings only; model, objective and optimizer settings are unchanged.
    env=os.environ.copy()
    env.pop('PYTHONPATH',None)
    env.pop('PYTHONHOME',None)
    env['OMP_NUM_THREADS']='4'
    env['MKL_NUM_THREADS']='4'
    run('anchored_gnn.train',['--graphs',graph_dir,'--targets',output/'targets.npz','--split',EXAMPLE/'split.csv','--config',ROOT/'configs/training.json','--seed',args.seed,'--output',output],env)
    run('anchored_gnn.predict',['--graphs',graph_dir,'--anchors',output/'test_anchors.npz','--checkpoint',output/'model_best.pt','--output',output/'predictions.npz'],env)
    run('anchored_gnn.evaluate',['--predictions',output/'predictions.npz','--targets',output/'targets.npz','--split',EXAMPLE/'split.csv','--group','test','--output',output/'test_metrics.json'],env)
    report(output)
    subprocess.run([sys.executable, '-E', '-B', str(EXAMPLE/'generate_figures.py'),
                    '--results', str(output), '--output', str(output/'figures')],
                   cwd=ROOT, env=env, check=True)
    print(f'Completed. Results: {output}',flush=True)


if __name__=='__main__':
    main()
