import unittest
import numpy as np
from src.revision.metrics import ranks_from_scores, summarize, random_metrics, bootstrap
from src.revision.common import split_for, cache_key, config


class RetrievalTests(unittest.TestCase):
    def test_perfect_and_wrong_pairs(self):
        scores = np.array([[5,2,1],[0,4,1],[0,1,3]])
        self.assertEqual(summarize(ranks_from_scores(scores))['mrr'],1)
        swapped=scores[:,[1,0,2]]
        np.testing.assert_array_equal(ranks_from_scores(swapped),[2,3,1])

    def test_invalid_scores_rejected(self):
        for x in [np.zeros((2,3)),np.array([[np.nan]])]:
            with self.assertRaises(ValueError): ranks_from_scores(x)

    def test_ties_do_not_all_hit(self):
        ranks=ranks_from_scores(np.ones((20,20)))
        self.assertEqual(summarize(ranks)['hit1_count'],1)
        self.assertEqual(summarize(ranks)['hit5_count'],5)

    def test_baseline_and_small_gallery(self):
        self.assertAlmostEqual(random_metrics(187)['hit5'],5/187)
        self.assertEqual(random_metrics(3)['hit5'],1)

    def test_bootstrap_reproducible(self):
        self.assertEqual(bootstrap([1,2,10],repeats=20),bootstrap([1,2,10],repeats=20))

    def test_split_and_cache(self):
        self.assertEqual(split_for('abcdefghijk'),split_for('abcdefghijk'))
        cfg=config();before=cache_key(cfg);cfg['frame_count']+=1
        self.assertNotEqual(before,cache_key(cfg))


if __name__=='__main__': unittest.main()
