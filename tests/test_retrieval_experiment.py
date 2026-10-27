import unittest
from retrieval_experiment import prompt_identity, response_status


class RetrievalRunTests(unittest.TestCase):
    def test_identical_conditions_share_identity_but_different_questions_do_not(self):
        a={"question_id":"q1","prompt":"same prompt","group":"D"}
        b=dict(a,group="H")
        config={"digest":"model1","num_ctx":8192}
        self.assertEqual(prompt_identity(a,config),prompt_identity(b,config))
        self.assertNotEqual(prompt_identity(a,config),prompt_identity(dict(a,prompt="changed"),config))
        self.assertNotEqual(prompt_identity(a,config),prompt_identity(dict(a,question_id="q2"),config))
        self.assertNotEqual(prompt_identity(a,config),prompt_identity(a,dict(config,digest="model2")))

    def test_missing_usage_and_nonfinal_responses_are_not_tokenizer_mismatches(self):
        empty={"raw_response":{"model":"","response":"","done":False},"token_count_match":False}
        self.assertEqual(response_status(empty),"incomplete_response")
        completed={"raw_response":{"model":"qwen2.5:3b","done":True,"prompt_eval_count":10,"eval_count":50},
                   "token_count_match":True}
        self.assertEqual(response_status(completed),"ok")
        completed["token_count_match"]=False
        self.assertEqual(response_status(completed),"tokenizer_mismatch")

    def test_length_terminated_but_complete_response_is_not_retried(self):
        r={"raw_response":{"model":"qwen2.5:3b","done":True,"done_reason":"length",
                           "prompt_eval_count":10,"eval_count":512},"token_count_match":True}
        self.assertEqual(response_status(r),"ok")


if __name__=="__main__": unittest.main()
