"""Метрики поиска исходной пары. Совместимость музыки оценивается отдельно."""
import numpy as np


def ranks_from_scores(scores):
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or not len(scores):
        raise ValueError('Expected nonempty square score matrix with paired diagonal')
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite scores')
    # Фиксированный tie-break, независимый от диагонали.
    permutation = np.random.default_rng(123).permutation(len(scores))
    order = permutation[np.argsort(-scores[:, permutation], axis=1, kind='stable')]
    return (order == np.arange(len(scores))[:, None]).argmax(1) + 1


def summarize(ranks):
    r = np.asarray(ranks)
    return dict(hit1=float(np.mean(r <= 1)), hit5=float(np.mean(r <= 5)),
                mrr=float(np.mean(1 / r)), mean_rank=float(r.mean()), n=len(r),
                hit1_count=int(np.sum(r <= 1)), hit5_count=int(np.sum(r <= 5)))


def random_metrics(n):
    if n < 1:
        raise ValueError('Empty gallery')
    return dict(hit1=1/n, hit5=min(5,n)/n, mrr=float(np.sum(1/np.arange(1,n+1))/n))


def bootstrap(ranks, seed=42, repeats=2000):
    # Интервалы по запросам условно на фиксированной галерее. Не покрывают все источники неопределённости.
    r = np.asarray(ranks)
    samples = np.random.default_rng(seed).choice(r, (repeats, len(r)), replace=True)
    result = {}
    for name, values in [('hit1', (samples <= 1).mean(1)), ('hit5', (samples <= 5).mean(1)),
                         ('mrr', (1 / samples).mean(1))]:
        result[name] = np.quantile(values, [.025, .975]).tolist()
    return result
