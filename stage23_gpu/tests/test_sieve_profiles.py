"""Empty/filtered profile catalogs produce a useful result, never a traceback."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import sieve_profiles


class ProfileLookup(unittest.TestCase):
    def test_empty_and_assumed_only(self):
        assumed = dict(id='assumed', digits=184, evidence='assumed')
        self.assertEqual(sieve_profiles.nearest([],184),[])
        self.assertEqual(sieve_profiles.nearest([assumed],184),[])
        self.assertEqual(sieve_profiles.nearest([assumed],184,True),[assumed])

    def test_ties_remain_available(self):
        profiles = [dict(id=str(n), digits=n, evidence='user') for n in (181,187,192)]
        self.assertEqual(sieve_profiles.nearest(profiles,184),profiles[:2])

    def test_cli_filtered_catalog(self):
        data = json.loads(sieve_profiles.CATALOG.read_text())
        data['profiles'] = data['profiles'][:1]
        data['profiles'][0]['evidence'] = 'assumed'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'profiles.json'
            path.write_text(json.dumps(data))
            args = [sys.executable,str(ROOT/'scripts/sieve_profiles.py'),'--catalog',str(path),'--digits','184']
            result = subprocess.run(args,capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('no eligible historical profiles',result.stderr)
            self.assertNotIn('Traceback',result.stderr)
            result = subprocess.run(args+['--include-assumed'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout),data['profiles'])


if __name__=='__main__':
    unittest.main()
