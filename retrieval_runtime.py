"""Isolate each unique request from a reproducible Ollama grammar-state fault."""
import time
from condition_model import ExperimentModel
from rag import _ollama_request, DEFAULT_LOCAL_MODEL


class RetrievalExperimentModel(ExperimentModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.metadata['request_isolation']='unload qwen2.5:3b before each unique generation, same weights/prompt/schema/options'

    def generate(self,prompt,max_new_tokens=512):
        start=time.perf_counter()
        reset=_ollama_request(self.base_url,'/api/generate',
                             {'model':DEFAULT_LOCAL_MODEL,'keep_alive':0,'stream':False},timeout=60)
        if reset.get('done') is not True:
            raise RuntimeError('Model unload did not complete; refusing a potentially contaminated request.')
        reset_seconds=time.perf_counter()-start
        result=super().generate(prompt,max_new_tokens)
        result['generation_request_seconds']=result['request_seconds']
        result['reset_seconds']=reset_seconds
        result['reset_response']=reset
        result['request_seconds']=time.perf_counter()-start
        return result
