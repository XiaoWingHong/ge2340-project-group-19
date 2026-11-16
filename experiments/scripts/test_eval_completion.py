#!/usr/bin/env python3
import json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import eval_three_arms as e
from cuda_logreg import ROOT
class Tests(unittest.TestCase):

 def test_missing_config_fails_before_data_or_network(self):
  with patch.dict(os.environ,{},clear=True),patch.object(sys,"argv",["eval","--env-file",str(ROOT/"experiments/tmp/no-such-env"),"--arms","B","--decision-backends","laya","--require-frozen-300"]),patch.object(e,"load_data") as load,patch("requests.post") as post,patch("requests.get") as get:
   with self.assertRaisesRegex(SystemExit,"LAYA_BASE_URL"):e.main()
   load.assert_not_called();post.assert_not_called();get.assert_not_called()
 def test_documented_noauth_convention(self):
  with patch.dict(os.environ,{"LAYA_BASE_URL":"http://127.0.0.1:8010","LAYA_MODEL":"english","LAYA_API_KEY":"none"},clear=True):
   c=e.resolve_decision_creds("laya");self.assertTrue(c["ready"]);self.assertEqual(c["api_key"],"");self.assertEqual(c["model"],"english")
 def test_skipped_and_partial_counts_never_mark_full_inference(self):
  self.assertFalse(e.completed_inference([]));self.assertFalse(e.completed_inference([{"status":"skipped","n_scored":0}]))
  a={"status":"partial","n_scored":300,"items":[{"task_id":i} for i in range(300)]};self.assertTrue(e.completed_inference([a]));a["items"].pop();self.assertFalse(e.completed_inference([a]))
 def test_runtime_endpoint_skip_writes_false_and_exits_failure(self):
  old={"arm":"B_laya","backend_id":"laya","status":"skipped","n_scored":0,"items":[],"notes":"fixture skipped"}
  with tempfile.TemporaryDirectory(dir=ROOT/"experiments/tmp") as temp:
   path=Path(temp)/"diagnostic.json"
   argv=["eval","--env-file",str(ROOT/"experiments/tmp/no-such-env"),"--arms","B","--decision-backends","laya","--require-frozen-300","--no-overwrite","--out-json",str(path)]
   with patch.dict(os.environ,{"LAYA_BASE_URL":"http://127.0.0.1:8010","LAYA_MODEL":"english","LAYA_API_KEY":"none"},clear=True),patch.object(sys,"argv",argv),patch.object(e,"run_decision_backend",return_value=old),patch("requests.post") as post,patch("requests.get") as get:
    with self.assertRaisesRegex(SystemExit,"incomplete"):e.main()
    post.assert_not_called();get.assert_not_called()
   p=json.loads(path.read_text())["provenance"];self.assertIs(p["inference_performed"],False);self.assertEqual(p["actual_inference_rows"],0)
if __name__=="__main__":unittest.main(verbosity=2)
