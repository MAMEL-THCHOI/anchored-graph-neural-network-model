"""Reproduce example parity and stress-locus figures from bundled results."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EX=ROOT/'examples'
sys.path.insert(0,str(ROOT))
from anchored_gnn.workflow import metric
from figure_style import configure_style,style_axes
from locus_fit import fit_yld2000,yld2000_curve

INK,RED,GRAY='#222222','#B7221A','#6E6E6E'
WORK=[.25,1.,2.,5.,10.]
REDS=['#E4B7B4','#D17D76','#B7221A','#942019','#65120E']
GRAYS=['white','#D7D7D7','#A0A0A0','#6E6E6E','#222222']
MARKERS=['o','^','s']
ANGLE_COLORS=[INK,'#A0A0A0',RED]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def table(path,fields,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(fields);w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=EX/'expected'/'seed7')
    parser.add_argument('--output',type=Path,default=ROOT/'figure_run')
    args=parser.parse_args()
    result=args.results.resolve();out=args.output.resolve()
    if out.exists(): raise FileExistsError('Use a new figure output directory.')
    out.mkdir(parents=True);data_out=out/'PLOTTED_DATA';data_out.mkdir()
    summary=json.loads((result/'summary.json').read_text())
    config=json.loads((ROOT/'configs/training.json').read_text())
    work=np.asarray(config['work_levels']);angles=np.arange(0,180,20)
    split={r['condition_id']:r['split'] for r in csv.DictReader((EX/'split.csv').open())}
    with np.load(EX/'targets.npz',allow_pickle=False) as target,np.load(result/'predictions.npz',allow_pickle=False) as pred:
        ids=pred['condition_ids'].tolist()
        assert len(ids)==3 and all(split[i]=='test' for i in ids)
        index=[target['condition_ids'].tolist().index(i) for i in ids]
        actual_h=np.asarray(target['hardening_radius'][index],float)
        actual_r=np.asarray(target['r_values'][index],float)
        predicted_h=np.asarray(pred['hardening_radius'],float)
        predicted_r=np.asarray(pred['r_values'],float)
    metrics={'hardening':metric(actual_h,predicted_h),'r':metric(actual_r,predicted_r)}
    recorded=json.loads((result/'test_metrics.json').read_text())
    for response in metrics:
        np.testing.assert_allclose(metrics[response]['R2'],recorded[response]['R2'],rtol=0,atol=1e-12)
    table(data_out/'parity_hardening.csv',['condition_id','work_mj_m3','loading_angle_deg','cpfem_mpa','prediction_mpa'],
          [[i,wp,angle,actual_h[n,w,a],predicted_h[n,w,a]] for n,i in enumerate(ids) for w,wp in enumerate(work) for a,angle in enumerate(angles)])
    table(data_out/'parity_lankford.csv',['condition_id','tensile_angle_deg','cpfem_r','prediction_r'],
          [[i,angle,actual_r[n,a],predicted_r[n,a]] for n,i in enumerate(ids) for a,angle in enumerate([0,45,90])])
    table(data_out/'parity_metrics.csv',['response','R2','MAE','MRE','rve_count','point_count'],
          [[name,m['R2'],m['mae'],m['mre'],len(ids),m['count']] for name,m in metrics.items()])
    components={(r['condition_id'],float(r['work_mj_m3']),float(r['loading_angle_deg'])):
                [float(r['sigma11_mpa']),float(r['sigma22_mpa'])]
                for r in csv.DictReader((EX/'cpfem_hardening.csv').open())}
    payload={};points=[];curves=[];fits=[];extent=0.
    for n,condition in enumerate(ids):
        payload[condition]=[]
        for stage,wp in enumerate(WORK):
            w=int(np.flatnonzero(np.isclose(work,wp))[0])
            reference=np.asarray([components[(condition,wp,float(a))]+[0.] for a in angles])
            # The existing manuscript display maps predicted radii to CPFEM normal-stress directions.
            scaled=reference*(predicted_h[n,w]/actual_h[n,w])[:,None]
            fit=fit_yld2000(scaled)
            assert fit['success']
            theta,x,y=yld2000_curve(fit)
            payload[condition].append((stage,reference,x,y))
            extent=max(extent,float(np.max(np.abs(reference[:,:2]))),float(np.max(np.abs(x))),float(np.max(np.abs(y))))
            fits.append({'condition_id':condition,'work_mj_m3':wp,**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fit.items()}})
            for a,angle in enumerate(angles):
                for sign in [1,-1]:
                    points.append([condition,wp,angle,sign<0,*list(sign*reference[a,:2]),*list(sign*scaled[a,:2]),actual_h[n,w,a],predicted_h[n,w,a]])
            curves.extend([[condition,wp,t,xx,yy] for t,xx,yy in zip(theta,x,y)])
    table(data_out/'locus_points.csv',['condition_id','work_mj_m3','loading_angle_deg','mirrored','cpfem_sigma11_mpa','cpfem_sigma22_mpa','scaled_sigma11_mpa','scaled_sigma22_mpa','cpfem_radius_mpa','predicted_radius_mpa'],points)
    table(data_out/'locus_fitted_curves.csv',['condition_id','work_mj_m3','polar_angle_rad','sigma11_mpa','sigma22_mpa'],curves)
    (data_out/'locus_fit_parameters.json').write_text(json.dumps(fits,indent=2)+'\n',encoding='utf-8')
    limit=100*math.ceil(1.04*extent/100)
    configure_style()

    def hardening(ax):
        for n,i in enumerate(ids): ax.scatter(actual_h[n],predicted_h[n],s=20,c=RED,marker=MARKERS[n],edgecolors=RED,linewidths=.6,alpha=.85)
        lo=25*math.floor(min(actual_h.min(),predicted_h.min())/25)
        hi=25*math.ceil(max(actual_h.max(),predicted_h.max())/25)
        ax.plot([lo,hi],[lo,hi],color=INK,lw=.9,ls='--',zorder=0)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel=r'CPFEM stress radius (MPa)',ylabel=r'Predicted stress radius (MPa)')
        ax.text(.06,.93,rf'$R^2={metrics["hardening"]["R2"]:.4f}$',transform=ax.transAxes,va='top')

    def lankford(ax):
        for n,i in enumerate(ids):
            for a,angle in enumerate([0,45,90]): ax.scatter(actual_r[n,a],predicted_r[n,a],s=42,c=ANGLE_COLORS[a],marker=MARKERS[n],edgecolors=ANGLE_COLORS[a],linewidths=.8)
        hi=math.ceil(max(actual_r.max(),predicted_r.max())*1.05)
        ax.plot([0,hi],[0,hi],color=INK,lw=.9,ls='--',zorder=0)
        ax.set(xlim=(0,hi),ylim=(0,hi),xlabel=r'CPFEM Lankford coefficient, $r$',ylabel=r'Predicted Lankford coefficient, $r$')
        ax.text(.06,.93,rf'$R^2={metrics["r"]["R2"]:.4f}$',transform=ax.transAxes,va='top')

    def locus(ax,condition):
        for stage,reference,x,y in payload[condition]:
            ax.plot(x,y,color=REDS[stage],lw=1.7,zorder=2)
            full=np.vstack([reference,-reference])
            ax.scatter(full[:,0],full[:,1],s=22,facecolors=GRAYS[stage],edgecolors=INK,lw=.65,zorder=3)
        ax.set(xlim=(-limit,limit),ylim=(-limit,limit),xlabel=r'$\sigma_{11}$ (MPa)',ylabel=r'$\sigma_{22}$ (MPa)')
        ax.text(.50,.50,condition,transform=ax.transAxes,ha='center',va='center',fontsize=11)

    manifests=[]
    def save(fig,stem,role,condition,source_csv):
        fig.canvas.draw()
        assert all(not ax.get_title() for ax in fig.axes)
        for suffix in ['.png','.svg']: fig.savefig(out/(stem+suffix),dpi=600,facecolor='white')
        with Image.open(out/(stem+'.png')) as im: width,height=im.size
        manifests.append(dict(stable_id=stem,scientific_role=role,condition_id=condition,domain='small-example internal test',
            png_path=(out/(stem+'.png')).relative_to(out.parents[1]).as_posix(),svg_path=(out/(stem+'.svg')).relative_to(out.parents[1]).as_posix(),
            svg_mode='true_vector',source_data_path='PLOTTED_DATA/'+source_csv,source_data_sha256=sha(data_out/source_csv),
            generator_path='examples/generate_figures.py',units='MPa or dimensionless r',uncertainty_definition='none; single seed',
            model_checkpoint=(result/'model_best.pt').relative_to(ROOT).as_posix() if (result/'model_best.pt').is_relative_to(ROOT) else 'model_best.pt',model_seed=summary['seed'],width_px=width,height_px=height,dpi=600))
        plt.close(fig)
    draws=[('hardening',hardening,'parity_hardening.csv',';'.join(ids)),('lankford',lankford,'parity_lankford.csv',';'.join(ids))]
    draws += [(f'locus_{i.lower()}',lambda ax,i=i:locus(ax,i),'locus_points.csv',i) for i in ids[:2]]
    for name,draw,source_csv,condition in draws:
        fig=plt.figure(figsize=(5,5));ax=fig.add_axes([.18,.16,.76,.76])
        draw(ax);style_axes(ax,square=True)
        save(fig,'example_panel_'+name,name,condition,source_csv)
    condition_handles=[Line2D([],[],marker=MARKERS[n],ls='none',color=INK,label=i,markersize=6) for n,i in enumerate(ids)]
    direction_handles=[Line2D([],[],marker='o',ls='none',color=c,label=f'{a}°',markersize=6) for a,c in zip([0,45,90],ANGLE_COLORS)]
    source_handles=[Line2D([],[],marker='o',ls='none',mfc='white',mec=INK,label='CPFEM',markersize=6),Line2D([],[],color=RED,lw=1.7,label='GNN-based Yld2000-2d fit')]
    work_handles=[Line2D([],[],marker='o',color=REDS[n],mfc=GRAYS[n],mec=INK,lw=1.6,label=rf'$W_p={wp:g}$',markersize=5) for n,wp in enumerate(WORK)]
    for name,handles in [('conditions',condition_handles),('tensile_directions',direction_handles),('locus_sources',source_handles),('plastic_work',work_handles)]:
        fig=plt.figure(figsize=(7.5,.65));fig.legend(handles=handles,loc='center',ncol=len(handles),frameon=False)
        save(fig,'example_legend_'+name,'legend','all','parity_metrics.csv')
    fig=plt.figure(figsize=(10.7,11.8))
    for n,(name,draw,source_csv,condition) in enumerate(draws):
        x=[.9,6.2][n%2];y=7.3 if n<2 else 1.55
        ax=fig.add_axes([x/10.7,y/11.8,3.8/10.7,3.8/11.8]);draw(ax);style_axes(ax,square=True)
        fig.text((x-.55)/10.7,(y+4.03)/11.8,f'({chr(97+n)})',fontsize=13,fontweight='bold')
    fig.legend(handles=condition_handles,loc='center',bbox_to_anchor=(.267,6.47/11.8),ncol=3,frameon=False,title='Test textures')
    fig.legend(handles=direction_handles,loc='center',bbox_to_anchor=(.76,6.47/11.8),ncol=3,frameon=False,title='Tensile direction')
    fig.legend(handles=source_handles,loc='center',bbox_to_anchor=(.5,.83/11.8),ncol=2,frameon=False)
    fig.legend(handles=work_handles,loc='center',bbox_to_anchor=(.5,.38/11.8),ncol=5,frameon=False)
    fig.text(.5,.05/11.8,r'Plastic work $W_p$ in MJ m$^{-3}$',ha='center',fontsize=9)
    save(fig,'example_panel_overview','four-panel overview',';'.join(ids),'parity_metrics.csv')
    table(out/'panel_export_manifest.csv',list(manifests[0]),[[r[k] for k in manifests[0]] for r in manifests])
    source_paths=[EX/'targets.npz',EX/'cpfem_hardening.csv',EX/'split.csv',result/'predictions.npz',result/'summary.json',result/'model_best.pt']
    source_manifest=[{'path':p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)} for p in source_paths]
    (out/'source_manifest.json').write_text(json.dumps(source_manifest,indent=2)+'\n',encoding='utf-8')
    (out/'CAPTIONS_EN.md').write_text(f'''# Example figure

**Figure. Prediction from a small CPFEM training example.** (a) Hardening stress-radius parity and (b) Lankford-coefficient parity for all three held-out test textures. Symbols identify the three RVEs; colors in (b) identify tensile directions. Dashed diagonals indicate equality. Both axes of each parity panel use identical limits and include every test point. (c,d) Normal-stress locus evolution for the first two test textures at plastic-work levels of 0.25, 1, 2, 5 and 10 MJ m⁻³. CPFEM normal-stress components are shown as symbols. Predicted stress radii are mapped onto the corresponding CPFEM normal-stress directions, following the study's existing display procedure, and red curves are Yld2000-2d fits to these mapped predictions. The fit uses an exponent of six. The opposite half-plane is obtained by central symmetry. These panels illustrate radial hardening prediction on supplied reference directions; they do not demonstrate an independent prediction of the stress direction or new loading-path CPFEM simulations. Fitting is used only for visualization and does not modify the parity data or model predictions.

All panels use the same single-seed example checkpoint (seed {summary['seed']}; selected epoch {summary['selected_epoch']}, training ended at epoch {summary['epochs_completed']}), trained on fourteen textures and selected using three validation textures. The test textures are {', '.join(ids)}; locus panels show {ids[0]} and {ids[1]}. The figure reports this small executable example, not the full-database paper performance. No seed uncertainty is shown.
''',encoding='utf-8')
    (out/'CAPTIONS_KO.md').write_text(f'''# 예제 그림 설명

(a) 세 시험 RVE의 hardening stress radius parity, (b) 같은 RVE의 Lankford coefficient parity입니다. 모든 시험값을 표시했으며 x·y축 범위를 동일하게 맞췄습니다. 모양은 RVE, (b)의 색은 인장 방향을 구분합니다.

(c,d)는 {ids[0]}과 {ids[1]}의 소성일 수준별 정상응력 평면 locus 변화입니다. CPFEM 결과를 점으로 표시하고, 예측한 stress radius를 해당 CPFEM 정상응력 방향에 배치한 뒤 기존 원고의 Yld2000-2d 표시 방식을 적용했습니다. 반대쪽 평면은 중심대칭으로 표시했습니다. 응력 방향 자체를 독립적으로 예측한 그림은 아니며, 시각화 피팅은 parity 계산이나 학습 결과를 바꾸지 않습니다.

모든 패널은 14개 학습 조건·3개 검증 조건으로 만든 seed {summary['seed']}의 {summary['selected_epoch']} epoch 가중치를 사용합니다. 학습은 {summary['epochs_completed']} epoch에서 종료됐습니다. 시험 조건은 3개이며, 3-seed 평균 또는 논문의 전체 학습 결과가 아닙니다.
''',encoding='utf-8')
    print(json.dumps({'metrics':metrics,'seed':summary['seed'],'selected_epoch':summary['selected_epoch'],'epochs_completed':summary['epochs_completed'],'output':str(out),'pairs':len(manifests)},indent=2))


if __name__=='__main__': main()
