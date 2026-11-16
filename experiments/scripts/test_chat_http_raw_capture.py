import json,unittest
from unittest.mock import Mock,patch
import eval_three_arms as e
class Tests(unittest.TestCase):
 def test_parser_input_unchanged_and_separate_reasoning_preserved(self):
  text='{"label":"admin","confidence":0.8}'
  body={"choices":[{"message":{"content":text,"reasoning_content":"Actual separate reasoning, not hidden"}}],"usage":{"completion_tokens_details":{"reasoning_tokens":37}}}
  r=Mock(ok=True,text=json.dumps(body));r.json.return_value=body
  with patch.object(e.requests,"post",return_value=r) as post:
   reply=e.chat_completion("http://local/v1","none","qwen",[])
  self.assertIsInstance(reply,str);self.assertEqual(str(reply),text);self.assertEqual(json.loads(reply.raw_http_body),body);self.assertEqual(e.parse_label_confidence(reply),("admin",0.8))
  payload=post.call_args.kwargs["json"];self.assertIs(payload["enable_thinking"],False);self.assertEqual(payload["temperature"],0);self.assertEqual(payload["max_tokens"],1024);self.assertNotIn("stop",payload)
 def test_reasoning_only_parse_failure_is_not_salvaged(self):
  body={"choices":[{"message":{"content":"","reasoning_content":"to=self long non-JSON reasoning"}}]}
  r=Mock(ok=True,text=json.dumps(body));r.json.return_value=body
  with patch.object(e.requests,"post",return_value=r):reply=e.chat_completion("http://local/v1","none","qwen",[])
  self.assertEqual(e.parse_label_confidence(reply),(None,None));self.assertEqual(json.loads(reply.raw_http_body),body)
if __name__=="__main__":unittest.main(verbosity=2)
