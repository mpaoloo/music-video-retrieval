"""Небольшие проекции поверх готовых признаков; пять заранее заданных опытов."""
import copy
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn import functional as F

from .common import ROOT, DATA, RUN, config, device, save_json, status, checkpoint_backup
from .metrics import ranks_from_scores, summarize, random_metrics, bootstrap

EXPERIMENTS = {
    'mel_all': ('mel', None),
    'ast_all': ('ast', None),
    'ast_audio_clean': ('ast', 'audio_clean'),
    'ast_pair_clean': ('ast', 'train_clean'),
    'ast_random_matched': ('ast', 'random_matched'),
}


class TwoTower(nn.Module):
    def __init__(self, video_dim, audio_dim, dim=64):
        super().__init__()
        self.video = nn.Linear(video_dim, dim)
        self.audio = nn.Linear(audio_dim, dim)

    def forward(self, video, audio):
        return F.normalize(self.video(video), dim=-1), F.normalize(self.audio(audio), dim=-1)


def normalize_fit(x):
    mean = x.mean(0)
    std = np.maximum(x.std(0), 1e-4)
    return mean.astype('float32'), std.astype('float32')


def standardize(x, stats):
    return ((x - stats[0]) / stats[1]).astype('float32')


def load_features(table, name):
    video, audio = [], []
    for path in table.feature_path:
        with np.load(ROOT / path) as f:
            video.append(f['video'])
            audio.append(f[name])
    return np.stack(video), np.stack(audio)


@torch.inference_mode()
def score(model, v, a, dev):
    model.eval()
    vz, az = model(torch.as_tensor(v, device=dev), torch.as_tensor(a, device=dev))
    return (vz @ az.T).cpu().numpy()


def fit(v, a, train_idx, val_idx, cfg, seed, dev):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    vs, aus = normalize_fit(v[train_idx]), normalize_fit(a[train_idx])
    v, a = standardize(v, vs), standardize(a, aus)
    model = TwoTower(v.shape[1], a.shape[1], cfg['embedding_dim']).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    vt = torch.as_tensor(v[train_idx], device=dev)
    at = torch.as_tensor(a[train_idx], device=dev)
    best, best_epoch, history, state = -1, 0, [], None
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        perm = torch.randperm(len(train_idx), device=dev)
        losses = []
        for idx in perm.split(cfg['batch_size']):
            if len(idx) < 2:
                continue
            vz, az = model(vt[idx], at[idx])
            logits = vz @ az.T / cfg['temperature']
            labels = torch.arange(len(idx), device=dev)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = summarize(ranks_from_scores(score(model, v[val_idx], a[val_idx], dev)))
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), **{'val_'+k: val for k,val in metrics.items()}))
        if metrics['mrr'] > best:
            best, best_epoch = metrics['mrr'], epoch
            state = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
        if epoch - best_epoch >= cfg['patience']:
            break
    model.load_state_dict(state)
    return model, v, a, vs, aus, history, best_epoch


def main():
    cfg = config()
    dev = device()
    torch.set_num_threads(4)
    table = pd.read_csv(DATA / 'features.csv', dtype={'pair_id':str})
    # Известный домен музыки фиксируется до сравнения моделей одинаковым правилом.
    val_idx = np.flatnonzero((table.split == 'val') & table.music_ok)
    test_idx = np.flatnonzero((table.split == 'test') & table.music_ok)
    if min(len(val_idx), len(test_idx)) < cfg['min_eval']:
        raise RuntimeError(f'Not enough music-domain eval data: val={len(val_idx)}, test={len(test_idx)}. Do not silently relax thresholds.')
    save_json(RUN / 'evaluation_protocol.json', dict(validation_ids=table.iloc[val_idx].video_id.tolist(),
              test_ids=table.iloc[test_idx].video_id.tolist(), fixed_music_gate=True,
              ground_truth='original paired audio; proxy, not subjective compatibility', config=cfg))
    results, validation, skipped = [], [], []
    fitted = []
    # Сначала обучение всех конфигураций и выбор по validation, затем test.
    for name, (audio_name, flag) in EXPERIMENTS.items():
        train_mask = table.split == 'train'
        if flag and flag != 'random_matched':
            train_mask &= table[flag]
        train_idx = np.flatnonzero(train_mask)
        if flag == 'random_matched':
            count = int(((table.split == 'train') & table.train_clean).sum())
            train_idx = np.sort(np.random.default_rng(cfg['seed']).choice(train_idx, count, replace=False))
        if len(train_idx) < cfg['min_train']:
            skipped.append(dict(experiment=name, reason='insufficient training pairs', n=len(train_idx)))
            continue
        v, a = load_features(table, audio_name)
        for seed in cfg['seeds']:
            status('training', experiment=name, seed=seed, train=len(train_idx), validation=len(val_idx))
            started = time.monotonic()
            model, sv, sa, vs, aus, history, best_epoch = fit(v, a, train_idx, val_idx, cfg, seed, dev)
            dest = RUN / name / str(seed)
            dest.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(history).to_csv(dest/'history.csv', index=False)
            torch.save(dict(state={k:x.detach().cpu() for k,x in model.state_dict().items()},
                            video_dim=v.shape[1], audio_dim=a.shape[1], dim=cfg['embedding_dim'],
                            video_mean=torch.from_numpy(vs[0]), video_std=torch.from_numpy(vs[1]),
                            audio_mean=torch.from_numpy(aus[0]), audio_std=torch.from_numpy(aus[1]),
                            audio_feature=audio_name, best_epoch=best_epoch, seed=seed, config=cfg), dest/'model.pt')
            val = summarize(ranks_from_scores(score(model, sv[val_idx], sa[val_idx], dev)))
            validation.append(dict(experiment=name, seed=seed, train_n=len(train_idx), best_epoch=best_epoch,
                                   seconds=time.monotonic()-started, **val))
            fitted.append((name,seed,audio_name,dest))
            pd.DataFrame(validation).to_csv(RUN/'validation.csv', index=False)
            checkpoint_backup()
    if not fitted:
        raise RuntimeError('No trainable experiments')
    averages = pd.DataFrame(validation).groupby('experiment').mrr.mean()
    winner = str(averages.idxmax())
    save_json(RUN/'selection.json', dict(experiment=winner, seed=cfg['seeds'][0],
              rule='highest mean validation MRR across fixed seeds; demo uses first seed'))
    for name, seed, audio_name, dest in fitted:
        model, vs, aus = load_model(dest/'model.pt', dev)
        v,a = load_features(table,audio_name)
        sv,sa = standardize(v,vs),standardize(a,aus)
        for domain, idx in [('music',test_idx), ('all',np.flatnonzero(table.split=='test'))]:
            scores = score(model,sv[idx],sa[idx],dev)
            ranks = ranks_from_scores(scores)
            m = summarize(ranks)
            results.append(dict(experiment=name,seed=seed,domain=domain,**m))
            np.save(dest/f'scores_{domain}.npy',scores)
            pd.DataFrame(dict(video_id=table.iloc[idx].video_id,rank=ranks)).to_csv(dest/f'ranks_{domain}.csv',index=False)
            save_json(dest/f'intervals_{domain}.json',bootstrap(ranks))
    # Дополнительный обучаемый baseline: ridge video -> AST, alpha зафиксирован заранее.
    v,a = load_features(table,'ast')
    tr=np.flatnonzero(table.split=='train')
    vs,aus=normalize_fit(v[tr]),normalize_fit(a[tr]);sv,sa=standardize(v,vs),standardize(a,aus)
    ridge=Ridge(alpha=10).fit(sv[tr],sa[tr])
    pred=ridge.predict(sv[test_idx]);target=sa[test_idx]
    pred/=np.maximum(np.linalg.norm(pred,axis=1,keepdims=True),1e-8)
    target=target/np.maximum(np.linalg.norm(target,axis=1,keepdims=True),1e-8)
    results.append(dict(experiment='ridge_ast',seed=42,domain='music',**summarize(ranks_from_scores(pred@target.T))))
    pd.DataFrame(results).to_csv(RUN/'test_metrics.csv',index=False)
    save_json(RUN/'random_baseline.json', {domain:dict(n=n,**random_metrics(n)) for domain,n in [('music',len(test_idx)),('all',int((table.split=='test').sum()))]})
    save_json(RUN/'skipped.json',skipped)
    status('training_complete', selected=winner, experiments=len(fitted),test_queries=len(test_idx))
    checkpoint_backup()


def load_model(path,dev):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=TwoTower(checkpoint['video_dim'],checkpoint['audio_dim'],checkpoint['dim']).to(dev)
    model.load_state_dict(checkpoint['state']);model.eval()
    vs=(checkpoint['video_mean'].numpy(),checkpoint['video_std'].numpy())
    aus=(checkpoint['audio_mean'].numpy(),checkpoint['audio_std'].numpy())
    return model,vs,aus


if __name__ == '__main__':
    main()
