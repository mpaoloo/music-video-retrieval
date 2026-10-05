"""Графики и воспроизводимая выборка для ручной оценки после обучения."""
import html
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ROOT, DATA, RUN, config, save_json, status


def main():
    cfg=config()
    table=pd.read_csv(DATA/'features.csv',dtype={'pair_id':str})
    protocol=json.loads((RUN/'evaluation_protocol.json').read_text())
    table=table.set_index('video_id').loc[protocol['test_ids']].reset_index()
    selection=json.loads((RUN/'selection.json').read_text())
    name=selection['experiment'];seed=selection['seed']
    scores=np.load(RUN/name/str(seed)/'scores_music.npy')
    rng=np.random.default_rng(2026)
    query_indices=rng.choice(len(table),min(12,len(table)),replace=False)
    review_dir=RUN/'review';media=review_dir/'media';media.mkdir(parents=True,exist_ok=True)
    queries=[];manifest=[]
    mel_path=RUN/'mel_all'/str(seed)/'scores_music.npy'
    mel=np.load(mel_path) if mel_path.exists() else None
    for i in query_indices:
        original=int(i);top=np.argsort(-scores[i])[:3].tolist()
        baseline=np.argsort(-mel[i])[:3].tolist() if mel is not None else []
        random_ids=rng.choice(len(table),min(3,len(table)),replace=False).tolist()
        candidates=sorted(set(top+baseline+random_ids+[original]));rng.shuffle(candidates)
        items=[]
        for j in candidates:
            row=table.iloc[j];vid=row.video_id
            dest=media/f'{vid}.mp4'
            if not dest.exists():shutil.copy2(ROOT/row.video_path,dest)
            item=dict(id=vid,src='media/'+vid+'.mp4',sources=[s for s,ok in [('model',j in top),('mel',j in baseline),('random',j in random_ids),('original',j==original)] if ok],
                      rank=int(np.where(np.argsort(-scores[i])==j)[0][0])+1)
            items.append(item)
            manifest.append(dict(query_id=table.iloc[i].video_id,candidate_id=vid,sources=';'.join(item['sources']),model_rank=item['rank'],compatibility='',comment=''))
        queries.append(dict(id=table.iloc[i].video_id,url=table.iloc[i].url,src='media/'+table.iloc[i].video_id+'.mp4',items=items))
    pd.DataFrame(manifest).to_csv(review_dir/'ratings_template.csv',index=False)
    save_json(review_dir/'selection.json',dict(seed=2026,queries=queries,note='Pooled candidates, not an exhaustive relevance annotation'))
    template=(ROOT/'templates/review.html').read_text()
    (review_dir/'index.html').write_text(template.replace('__QUERIES__',json.dumps(queries,ensure_ascii=False).replace('</','<\\/')))
    metrics=pd.read_csv(RUN/'test_metrics.csv')
    music=metrics[metrics.domain=='music']
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    baseline=json.loads((RUN/'random_baseline.json').read_text())['music']
    for ax,key in zip(axes,['hit1','hit5','mrr']):
        summary=music.groupby('experiment')[key].agg(['mean','std']).fillna(0)
        ax.barh(summary.index,summary['mean'],xerr=summary['std'])
        ax.axvline(baseline[key],color='black',linestyle='--',label='Random expectation')
        ax.set_xlabel(key+' (fraction)');ax.set_title(key+f"; {baseline['n']} candidates")
        ax.legend(fontsize=8)
    fig.suptitle('Test: same music-domain gallery; mean ± std across training seeds')
    fig.tight_layout();fig.savefig(RUN/'test_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for exp in music.experiment.unique():
        path=RUN/exp/str(seed)/'history.csv'
        if not path.exists():continue
        h=pd.read_csv(path)
        axes[0].plot(h.epoch,h.train_loss,label=exp);axes[1].plot(h.epoch,h.val_mrr,label=exp)
    for ax,y in zip(axes,['Training cross entropy','Validation MRR']):
        ax.set_xlabel('Epoch');ax.set_ylabel(y);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(RUN/'training.png',dpi=160);plt.close(fig)
    status('review_ready',queries=len(queries),path=str(review_dir/'index.html'))


if __name__=='__main__':main()
