"""Сводка ручной оценки на одинаковых полностью размеченных запросах."""
import argparse
import json

import pandas as pd

from .common import RUN


def summarize_ratings(ratings, manifest):
    keys = ['query_id', 'candidate_id']
    if ratings.duplicated(keys).any():
        raise ValueError('Duplicate rated pairs')
    ratings = ratings.copy()
    ratings['compatibility'] = pd.to_numeric(ratings.compatibility, errors='raise')
    if not ratings.compatibility.dropna().isin([0, 1, 2]).all():
        raise ValueError('Expected ratings 0, 1 or 2')
    unknown = ratings.merge(manifest[keys], on=keys, how='left', indicator=True)
    if not unknown['_merge'].eq('both').all():
        raise ValueError('Pair not found in review manifest')
    merged = manifest.drop(columns=['compatibility', 'comment']).merge(
        ratings, on=keys, how='left', validate='one_to_one')
    complete = [query for query, group in merged.groupby('query_id')
                if group.compatibility.notna().all()]
    rows = []
    for source in ['model', 'mel', 'random', 'original']:
        group = merged[merged.query_id.isin(complete) &
                       merged.sources.str.split(';').apply(lambda values: source in values)]
        by_query = group.groupby('query_id').compatibility.mean()
        rows.append(dict(source=source, queries=len(by_query), pairs=len(group),
                         mean_rating=float(by_query.mean()),
                         good_fraction=float(group.compatibility.eq(2).mean())))
    top1 = merged[merged.query_id.isin(complete) & (merged.model_rank == 1)]
    details = dict(scored_pairs=int(ratings.compatibility.notna().sum()),
                   scored_queries=int(ratings[ratings.compatibility.notna()].query_id.nunique()),
                   complete_queries=complete,
                   complete_pairs=int(merged[merged.query_id.isin(complete)].shape[0]),
                   top1_mean=float(top1.compatibility.mean()),
                   top1_good_count=int(top1.compatibility.eq(2).sum()),
                   top1_acceptable_count=int(top1.compatibility.ge(1).sum()),
                   note='Single author; ordinal scores summarized descriptively. Missing scores remain missing.')
    return pd.DataFrame(rows), details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratings', default=str(RUN / 'review/human_ratings.csv'))
    args = parser.parse_args()
    manifest = pd.read_csv(RUN / 'review/ratings_template.csv')
    ratings = pd.read_csv(args.ratings)
    summary, details = summarize_ratings(ratings, manifest)
    summary.to_csv(RUN / 'review/human_summary.csv', index=False)
    (RUN / 'review/human_summary.json').write_text(json.dumps(details, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
