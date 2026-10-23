"""CPU tests for independent reconstruction and missing/censored provenance."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'stage23_bench/tools'))
from stage1_provenance import audit, reconstruct, join_raw, join_sopt, join_ropt, integer_root
from polyfmt import cado_poly_text, translate, coeffs

class Provenance(unittest.TestCase):
    def setUp(self):
        self.f=json.loads((Path(__file__).parent/'data/stage1_hit.json').read_text())
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'audit.jsonl'
    def rows(self):
        f=self.f; h=dict(f['hit'],batch=1); b=dict(f['batch'],batch=1)
        return [f['run'],f['coefficient'],b,h,
            dict(type='batch_end',coefficient=h['coefficient'],batch=h['batch'],returned=1,saturated=False),
            dict(type='coefficient_end',coefficient=h['coefficient'],complete=True),
            dict(type='worker_end')]
    def write(self,rows):
        self.path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return audit([self.path])
    def test_integer_root(self):
        for d in (4,5,6):
            for x in (2,17,2**100+7):
                for n in (x**d-1,x**d,x**d+1):
                    r=integer_root(n,d); self.assertTrue(r**d<=n<(r+1)**d)
    def test_reconstruction_and_raw_sopt_identity(self):
        ledger,counts,_,incomplete=self.write(self.rows())
        self.assertEqual(counts['valid_hits'],1); self.assertFalse(incomplete)
        rawpath=Path(self.tmp.name)/'raw.poly'
        rawpath.write_text(cado_poly_text(self.f['raw']))
        self.assertEqual(join_raw(ledger,[rawpath]),0)
        # Sopt may change sign/multiplier and translate its rational polynomial.
        raw=self.f['raw']; opt=copy.deepcopy(raw)
        for i,c in enumerate(translate(coeffs(raw),1)):
            opt[f'c{i}']=-3*c
        opt['Y0']+=opt['Y1']
        sop=Path(self.tmp.name)/'sopt.out'
        sop.write_text('### Input raw polynomial\n'+''.join('# '+l+'\n' for l in cado_poly_text(raw).splitlines())+
            '### Size-optimized polynomial\n'+cado_poly_text(opt)+
            '# lognorm 1.0, exp_E 2.0, alpha -1.0 (proj -0.1), 1 real\n')
        self.assertEqual(join_sopt(ledger,[sop]),0)
        self.assertEqual(len(next(iter(ledger.values()))['sopt']),1)
        ropt=copy.deepcopy(opt)
        for i,c in enumerate(translate(coeffs(opt),2)):
            ropt[f'c{i}']=-c
        ropt['Y0']+=2*ropt['Y1']
        ropt['c0']+=7*ropt['Y0']; ropt['c1']+=7*ropt['Y1']
        rp=Path(self.tmp.name)/'ropt.poly'; rp.write_text(cado_poly_text(ropt))
        self.assertEqual(join_ropt(ledger,[rp],'poly'),0)
        self.assertEqual(len(next(iter(ledger.values()))['ropt']),1)
        ropt['c0']+=1; rp.write_text(cado_poly_text(ropt))
        self.assertEqual(join_ropt(ledger,[rp],'poly'),1)
    def test_rejected_hit_replays_without_changing_identity(self):
        rows=self.rows(); rows[3]=dict(rows[3],admitted=0,valid=None)
        ledger,counts,_,_=self.write(rows)
        self.assertEqual(counts['rejected_hits'],1); self.assertEqual(len(ledger),1)
    def test_corrupt_modular_condition(self):
        h=dict(self.f['hit'],offset=str(int(self.f['hit']['offset'])+1))
        self.assertIsNone(reconstruct(self.f['run'],self.f['coefficient'],h))
    def test_truncated_and_saturated(self):
        rows=self.rows(); self.assertTrue(self.write(rows[:-3])[3])
        rows[4]['saturated']=True; self.assertTrue(self.write(rows)[3])
    def test_duplicate_batch_rejected(self):
        rows=self.rows(); rows.insert(3,rows[2])
        with self.assertRaises(ValueError): self.write(rows)
    def test_streaming_matches_exact_and_retires_batches(self):
        rows=self.rows()
        rows.insert(5,dict(rows[2],batch=2))
        rows.insert(6,dict(rows[4],batch=2,returned=0))
        self.write(rows)
        exact=audit([self.path],streaming=False)
        summary={}
        streamed=audit([self.path],streaming=True,summary=summary)
        self.assertEqual(exact[0],streamed[0])
        self.assertFalse(streamed[3])
        self.assertIsNone(streamed[1]['repeated_qroot_work'])
        self.assertEqual(sum(exact[2].values()),sum(streamed[2].values()))
        self.assertEqual(sum(e['qroots'] for e in summary['exposure']),exact[1]['qroots'])
        rows.insert(7,rows[2])  # replay of a retired batch is still rejected
        self.path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(ValueError): audit([self.path],streaming=True)
    def test_streaming_incomplete_and_totals(self):
        self.write(self.rows()[:-3])
        self.assertTrue(audit([self.path],streaming=True)[3])
        rows=self.rows()
        rows[-2]['qroots_done']=0
        with self.assertRaises(ValueError): self.write(rows)
        rows=self.rows()
        rows.insert(4,dict(type='engine',coefficient=1,batch=1,hash_caps=1))
        self.assertTrue(self.write(rows)[3])
    def test_truncated_tail_keeps_valid_candidates(self):
        exact=self.write(self.rows()[:-2])
        with self.path.open('a') as f: f.write('{"type":"coefficient_end","coefficient":')
        result=audit([self.path])
        self.assertEqual(exact[0],result[0])
        self.assertTrue(any('truncated final' in s for s in result[3]))
        with self.path.open('a') as f: f.write('\n')
        with self.assertRaises(ValueError): audit([self.path])
    def test_launch_changes_count_every_completed_batch(self):
        rows=self.rows()
        rows[0]=dict(rows[0],schema=2,launch_records='changes')
        shape=dict(type='launch',coefficient=1,batch=1,group=0,num_p=10,
                   roots_per_p=1,blocks_x=1,blocks_y=1,threads=32,q_tile=2)
        rows.insert(3,shape)
        rows.insert(6,dict(rows[2],batch=2))
        rows.insert(7,dict(rows[5],batch=2,returned=0))
        self.write(rows)
        summary={}
        audit([self.path],summary=summary)
        self.assertEqual(summary['launches'][0]['calls'],2)
    def test_existing_ledger_fails_before_input_read(self):
        import subprocess
        ledger=Path(self.tmp.name)/'exists'; ledger.write_text('preserve')
        result=subprocess.run([sys.executable,str(ROOT/'stage23_bench/tools/stage1_provenance.py'),
            str(Path(self.tmp.name)/'nonexistent-input'),'--ledger',str(ledger)],
            capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('FileExistsError',result.stderr)
        self.assertEqual(ledger.read_text(),'preserve')
    def test_compressed_sidecar_and_log_exposure(self):
        import subprocess
        plain=self.write(self.rows())
        compressed=Path(str(self.path)+'.zst')
        subprocess.run(['zstd','-q',str(self.path),'-o',str(compressed)],check=True)
        self.path.unlink()
        summary={}
        packed=audit([self.path],summary=summary)
        self.assertEqual(plain,packed)
        self.assertEqual(sum(r['qroots'] for r in summary['log2_exposure']),packed[1]['qroots'])
        for row in summary['log2_exposure']:
            self.assertEqual(row['q_min'],1<<row['log2_bin'])
            self.assertEqual(row['progression_qroots'],row['qroots']*row['aprog'])
        compressed.write_bytes(compressed.read_bytes()[:-5])
        with self.assertRaises(OSError): audit([compressed])
    def test_virtual_archive_extract(self):
        import subprocess
        from unittest.mock import patch
        import sidecar_io
        source=Path(self.tmp.name)/'source'
        data=b'header\nignore me\n'+b'coefficient\nend\n'
        source.write_bytes(data)
        archive=Path(str(source)+'.zst')
        subprocess.run(['zstd','-q',str(source),'-o',str(archive)],check=True)
        target=Path(self.tmp.name)/'missing.jsonl'
        registry=Path(self.tmp.name)/'registry.json'
        registry.write_text(json.dumps({'extracts': {str(target):
            {'archive':str(archive),'ranges':[[0,7],[17,len(data)]]}}}))
        with patch.object(sidecar_io,'REGISTRY',registry):
            self.assertEqual(b''.join(sidecar_io.chunks(target)),b'header\ncoefficient\nend\n')

if __name__=='__main__': unittest.main()
