"""Declared-skew controls must survive proposal screening and final validation."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'stage23_bench/tools'))
import final_scoring as final


class FinalScoring(unittest.TestCase):
    def setUp(self):
        self.poly = dict(n=17, Y0=-2, Y1=1, c0=19, c1=-1, skew=100.)
        self.args = SimpleNamespace(params='job:29,30,14,10e6', useskew=False,
                                    latseed=0, points=4000, eval_points=16000)

    def murphy(self, polys, n, args):
        if not args.useskew:
            return [(12., 200., 0) for p in polys]
        return [({100.:10., 200.:12., 300.:11.}[p['skew']], p['skew'], 0) for p in polys]

    def lattice(self, polys, n, args):
        if not args.useskew:
            return [(13., 300., 10., [1., 2.]) for p in polys]
        table = {100.:10., 200.:12., 300.:13.} if args.latseed == 0 else {100.:20., 200.:12., 300.:11.}
        return [(table[p['skew']], p['skew'], 10., [1., 2.]) for p in polys]

    def test_declared_control_survives_screen_reversal(self):
        original = copy.deepcopy(self.args)
        with patch.object(final.scores, 'murphy', self.murphy), patch.object(final.scores, 'lattice_scores', self.lattice):
            screened = final.score_candidates([dict(poly=self.poly)], 17, self.args, propose=True)
            self.assertEqual(screened[0]['lattice_skew'], 300.)
            self.assertEqual(screened[0]['skew_candidates'], [100., 200., 300.])
            self.args.latseed = 2
            checked = final.score_candidates(screened, 17, self.args)
        self.assertEqual(checked[0]['lattice_skew'], 100.)
        self.assertEqual(checked[0]['lattice_score'], 20.)
        self.assertEqual(checked[0]['standard_skew'], 200.)
        self.assertEqual(checked[0]['declared_scores']['lattice_score'], 20.)
        self.assertEqual(final.winner_configuration(checked[0], 'lattice_score')['poly']['skew'], 100.)
        self.assertEqual(final.winner_configuration(checked[0], 'standard_murphy')['poly']['skew'], 200.)
        self.assertEqual(self.args.params, original.params)
        self.assertEqual(self.args.useskew, original.useskew)

    def test_same_coefficients_keep_each_declared_skew(self):
        rows = [dict(poly=self.poly), dict(poly=dict(self.poly, skew=300.))]
        with patch.object(final.scores, 'murphy', self.murphy), patch.object(final.scores, 'lattice_scores', self.lattice):
            checked = final.score_candidates(rows, 17, self.args, propose=True)
        self.assertEqual(checked[0]['declared_scores']['skew'], 100.)
        self.assertEqual(checked[1]['declared_scores']['skew'], 300.)
        self.assertEqual(checked[0]['declared_scores']['lattice_score'], 10.)
        self.assertEqual(checked[1]['declared_scores']['lattice_score'], 13.)

    def test_missing_skew_gets_proposals_and_invalid_skew_is_rejected(self):
        poly = dict(self.poly)
        del poly['skew']
        with patch.object(final.scores, 'murphy', self.murphy), patch.object(final.scores, 'lattice_scores', self.lattice):
            row = final.score_candidates([dict(poly=poly)], 17, self.args, propose=True)[0]
            self.assertIsNone(row['declared_scores'])
            self.assertEqual(row['skew_candidates'], [200., 300.])
            for skew in (0., -1., float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    final.score_candidates([dict(poly=dict(poly, skew=skew))], 17, self.args, propose=True)

    def test_content_candidate_survives_low_printed_rank(self):
        def block(e, cs, skew=100.):
            return (f'# norm 1 alpha -1 e {e} rroots 1\nskew: {skew}\n' +
                    ''.join(f'c{i}: {c}\n' for i,c in enumerate(cs)) + 'Y0: -2\nY1: 1\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'msieve.p'
            # Repeated exports must not consume the three distinct leader slots.
            path.write_text(block(10., [13, 0, 1])*4 + block(9., [11, 1, 1]) +
                            block(8., [9, 2, 1]) + block(7., [7, 3, 1]) +
                            block(1., [10, 8, 2], 123.456)*2)
            rows = final.msieve_shortlist(path, 17)
            self.assertEqual(len(rows), 4)
            self.assertEqual(sorted(r['reported_murphy'] for r in rows), [1., 8., 9., 10.])
            reduced = next(r for r in rows if r['source_content'] == 2)
            self.assertEqual(reduced['index'], 7)
            self.assertEqual(reduced['poly'], dict(n=17, Y0=-2, Y1=1, c0=5, c1=4, c2=1, skew=123.456))
            path.write_text(block(1., [17, 0, 17]))
            with self.assertRaisesRegex(ValueError, 'shares a factor'):
                final.msieve_shortlist(path, 17)

    def test_all_producers_score_primitive_coefficients(self):
        original = dict(self.poly, c0=38, c1=-2, lognorm=999.)
        rows = [dict(method=method, poly=original) for method in ('msieve_orig', 'cado_orig', 'gpu_pipeline')]

        def murphy(polys, n, args):
            self.assertTrue(all(p['c0'] == 19 and p['c1'] == -1 and 'lognorm' not in p for p in polys))
            return self.murphy(polys, n, args)

        def lattice(polys, n, args):
            self.assertTrue(all(p['c0'] == 19 and p['c1'] == -1 for p in polys))
            return self.lattice(polys, n, args)

        with patch.object(final.scores, 'murphy', murphy), patch.object(final.scores, 'lattice_scores', lattice):
            screened = final.score_candidates(rows, 17, self.args, propose=True)
            checked = final.score_candidates(screened, 17, self.args)
        self.assertTrue(all(r['source_content'] == 2 and r['poly'] == self.poly for r in checked))
        self.assertEqual(original['c0'], 38)
        self.assertEqual(original['lognorm'], 999.)
        for method in ('cado_orig', 'gpu_pipeline'):
            with self.assertRaisesRegex(ValueError, 'shares a factor'):
                final.score_candidates([dict(method=method, poly=dict(self.poly, c0=17, c1=17))], 17, self.args)

    def test_fixed_skew_precision_and_legacy_rendering(self):
        skew = 123.45678901234567
        self.assertEqual(final.scores.scored_skew(format(skew, '.17g'), skew), skew)
        self.assertEqual(final.scores.scored_skew(format(skew, '.1f'), skew), skew)
        # The old .051 tolerance accepted an incorrect value this close.
        for token in ('123.457', '123.45', 'nan', 'inf'):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, 'changed the declared skew'):
                final.scores.scored_skew(token, skew)


if __name__ == '__main__':
    unittest.main()
