import json
from types import SimpleNamespace
import unittest

from hybrid_retrieval import reciprocal_rank_fusion
from hybrid_runtime import display_answer, run_plan_query


class RuntimeTests(unittest.TestCase):
    def test_bad_json_and_refusal_cannot_become_operation_instructions(self):
        self.assertIsNone(display_answer('{"short_answer":"rm', [])['answer'])
        raw=json.dumps(dict(short_answer='do something', explanation='缺少证据', citations=[], abstain=True))
        result=display_answer(raw, [])
        self.assertEqual(result['status'], 'abstained')
        self.assertNotIn('do something', result['answer'])

    def test_invented_citation_is_reported_and_not_rendered_as_a_source(self):
        raw=json.dumps(dict(short_answer='answer', explanation='explain', citations=['P1','P99'], abstain=False))
        result=display_answer(raw,[{'id':'one'}])
        self.assertEqual(result['invalid_citations'],['P99'])
        self.assertIn('[P1]',result['answer'])
        self.assertNotIn('[P99]',result['answer'])

    def test_input_budget_failure_never_calls_generator(self):
        p=dict(id='a', title='Guide',text='x'*200,scope='',score=1.0)
        rankings={'dense':[p],'bm25':[p],'fused':reciprocal_rank_fusion([p],[p],[p]),
                  'timings':{'total_seconds':.001}}
        retriever=SimpleNamespace(passages=[p],dense=SimpleNamespace(),search_all=lambda q:rankings)
        def forbidden(_): raise AssertionError('must not generate over budget')
        runtime=SimpleNamespace(backend=None,model_id=None,num_ctx=512,count=len,
            prompt=lambda q,ps:q+''.join(p['text'] for p in ps),generate=forbidden)
        result=run_plan_query(retriever,'question','X',runtime,generate=True)
        self.assertEqual(result['status'],'input_budget_exceeded')
        self.assertIsNone(result['answer'])

    def test_empty_direct_answer_is_not_shown_as_success(self):
        raw=json.dumps(dict(short_answer='',explanation='some explanation',citations=['P1'],abstain=False))
        self.assertEqual(display_answer(raw,[{}])['status'],'empty_answer')


if __name__=='__main__': unittest.main()
