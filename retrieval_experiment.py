"""Frozen D/H/X retrieval study; old data are regression, new tasks are exploratory."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
from statistics import mean, median
import subprocess
import time

from condition_experiment import (now, read_json, write_json, read_jsonl, write_jsonl,
    append_jsonl, public_question, retrieval_query, validate_gold, unique_records,
    validate_schedule, capture_host, write_csv, apply_review_overrides)
from condition_model import QwenTokenizer, make_prompt, sha256_file, SCHEMA
from retrieval_runtime import RetrievalExperimentModel as ExperimentModel
from condition_metrics import score_answer, aggregate_review
from hybrid_retrieval import HybridRetriever, build_contexts

ROOT = Path(__file__).parent
DATA = ROOT / "data/retrieval_eval_v2"
OLD = ROOT / "runs/condition_eval/2026-10-27_test_v1"
GROUPS = ("D", "H", "X")


def prompt_identity(item, model_metadata):
    payload = [item["question_id"], item["prompt"], model_metadata, SCHEMA, 512]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def response_status(record):
    raw = record.get("raw_response")
    if (not isinstance(raw, dict) or raw.get("done") is not True or not raw.get("model")
            or type(raw.get("prompt_eval_count")) is not int or type(raw.get("eval_count")) is not int):
        return "incomplete_response"
    return "ok" if record.get("token_count_match") is True else "tokenizer_mismatch"


def prepare(args):
    folder = Path(args.run_dir)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Use a new empty run directory; existing records are never overwritten.")
    if args.split == "development":
        content = read_json(ROOT / "data/condition_eval/development.json")
        source = content.get("questions", content.get("items"))
        questions = [public_question(q) for q in source]
        gold = {"metadata": {"split": "development"}, "items": source}
    elif args.split == "regression":
        questions = read_json(OLD / "questions.json")
        gold = read_json(OLD / "gold.json")
    else:
        questions = read_json(DATA / "holdout_questions.json")["questions"]
        gold = read_json(DATA / "holdout_gold.json")
    validate_gold(questions, gold)
    corpus = read_json(OLD / "corpus.json")
    byid = {p["id"]: p for p in corpus["passages"]}
    for g in gold["items"]:
        for e in g["relevant_evidence"]:
            assert e["quote"] in byid[e["passage_id"]]["text"], (g["id"], e["passage_id"])
    tokenizer = QwenTokenizer()
    retriever = HybridRetriever(corpus["passages"])
    model = ExperimentModel(tokenizer)
    records, retrievals = [], []
    for question in questions:
        q = public_question(question)
        query = retrieval_query(q["question"])
        rankings = retriever.search_all(query)
        retrievals.append({"question_id": q["id"], "query": query, **rankings})
        contexts = build_contexts(rankings, corpus["passages"], tokenizer.count,
                                  lambda ps: make_prompt(q["question"], ps))
        for group in GROUPS:
            row = dict(contexts[group], question_id=q["id"], question=q,
                       retrieval_query=query, key=q["id"]+":"+group)
            row["prompt_identity"] = prompt_identity(row, model.metadata)
            records.append(row)
    ordered = list(questions)
    rng = random.Random(42)
    rng.shuffle(ordered)
    schedule = []
    for q in ordered:
        groups = list(GROUPS)
        rng.shuffle(groups)
        schedule.extend(q["id"]+":"+g for g in groups)
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in [("corpus.json",corpus),("questions.json",questions),("gold.json",gold),
                        ("schedule.json",schedule)]:
        write_json(folder/name,value)
    write_jsonl(folder/"inputs.jsonl",records)
    write_jsonl(folder/"retrievals.jsonl",retrievals)
    source_paths = list(ROOT.glob("*.py")) + list((ROOT/"tests").glob("test_*condition*.py"))
    source_paths += list((ROOT/"tests").glob("test_*retrieval*.py"))
    source_paths += list(DATA.glob("*.json"))+list(DATA.glob("*.md"))
    source_paths += [ROOT/"requirements.txt", ROOT/"data/condition_eval/development.json"]
    source_hashes = {}
    for path in sorted(set(source_paths)):
        relative = path.relative_to(ROOT).as_posix()
        destination = folder/"source_snapshot"/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        source_hashes[relative] = sha256_file(path)
    manifest = {
        "prepared_at":now(),"split":args.split,"n_questions":len(questions),
        "planned_positions":len(schedule),"unique_prompt_conditions":len({r["prompt_identity"] for r in records}),
        "model":model.metadata,"retrieval_config":retrievals[0]["config"],
        "protocol":{"version":"retrieval-v2","groups":list(GROUPS),"seed":42,"max_new_tokens":512,
                    "num_ctx":8192,"extension_budget":512,"deduplicate_identical_prompts_within_task":True,
                    "runtime_retries":1,"no_semantic_or_format_retries":True,
                    "old_tasks_are_exposed_regression":True,"new_tasks_are_exploratory":True,
                    "review_type":"ai_assisted","human_review_completed":False},
        "source_hashes":source_hashes,
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "git_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True).strip(),
        "file_hashes":{name:sha256_file(folder/name) for name in
                       ["corpus.json","questions.json","gold.json","schedule.json","inputs.jsonl","retrievals.jsonl"]},
    }
    write_json(folder/"manifest.json",manifest)
    write_json(folder/"host_before.json",capture_host())
    print(f"Prepared {args.split}: {len(questions)} tasks, {len(schedule)} positions, "
          f"{manifest['unique_prompt_conditions']} unique prompt conditions.",flush=True)


def verify(folder):
    m=read_json(folder/"manifest.json")
    for name,digest in m["file_hashes"].items():
        if sha256_file(folder/name)!=digest: raise ValueError("Frozen input changed: "+name)
    for name,digest in m["source_hashes"].items():
        if sha256_file(ROOT/name)!=digest: raise ValueError("Frozen source changed: "+name)
    return m


def run(args):
    folder=Path(args.run_dir)
    m=verify(folder)
    inputs=unique_records(read_jsonl(folder/"inputs.jsonl"),"inputs")
    schedule=read_json(folder/"schedule.json")
    validate_schedule(schedule,inputs)
    already=unique_records(read_jsonl(folder/"predictions.jsonl"),"predictions",set(inputs))
    if any(r["status"]=="tokenizer_mismatch" for r in already.values()):
        raise RuntimeError("Unresolved completed-response token mismatch; cannot resume automatically.")
    canonical={}
    for key,r in already.items():
        canonical.setdefault(inputs[key]["prompt_identity"],r)
    model=ExperimentModel(QwenTokenizer())
    if model.metadata != m["model"]: raise ValueError("Model configuration changed.")
    append_jsonl(folder/"warmups.jsonl",{"at":now(),"checks":model.validate_tokenizer()})
    for position,key in enumerate(schedule,1):
        if key in already: continue
        item=inputs[key];identity=item["prompt_identity"]
        result={"key":key,"question_id":item["question_id"],"group":item["group"],
                "position":position,"recorded_at":now(),"attempts":0,
                "prompt_identity":identity,"reused_from":None,"request_seconds":None}
        if identity in canonical:
            previous=canonical[identity]
            for field in ["raw_response","status","token_count_match","expected_input_tokens","error"]:
                if field in previous: result[field]=previous[field]
            result["reused_from"]=previous["reused_from"] or previous["key"]
        elif item["status"]!="ready":
            result.update(status=item["status"],raw_response=None)
        else:
            for attempt in (1,2):
                result["attempts"]=attempt;start=time.perf_counter()
                try:
                    response=model.generate(item["prompt"],512)
                    status=response_status(response)
                    append_jsonl(folder/"attempts.jsonl",{"key":key,"attempt":attempt,
                        "recorded_at":now(),"classified_status":status,**response})
                    result.update(response,status=status)
                    if status in ("ok","tokenizer_mismatch"): break
                except RuntimeError as exc:
                    result.update(status="runtime_error",raw_response=None,error=str(exc),
                                  token_count_match=None,request_seconds=time.perf_counter()-start)
                    append_jsonl(folder/"attempts.jsonl",{"key":key,"attempt":attempt,
                        "recorded_at":now(),"error":str(exc),"classified_status":"runtime_error",
                        "request_seconds":result["request_seconds"]})
                if attempt==1: time.sleep(1)
        append_jsonl(folder/"predictions.jsonl",result)
        already[key]=result;canonical.setdefault(identity,result)
        reuse=f" reused {result['reused_from']}" if result["reused_from"] else ""
        print(f"{position}/{len(schedule)} {key}: {result['status']}{reuse}",flush=True)
        if result["status"]=="tokenizer_mismatch":
            raise RuntimeError("Completed-response tokenizer mismatch; raw data saved, run stopped.")
    write_json(folder/"host_after.json",capture_host())


def analyze(args):
    folder=Path(args.run_dir);m=verify(folder)
    inputs=unique_records(read_jsonl(folder/"inputs.jsonl"),"inputs")
    predictions=unique_records(read_jsonl(folder/"predictions.jsonl"),"predictions",set(inputs))
    if set(predictions)!=set(inputs): raise ValueError("Incomplete run, denominators must not change.")
    gold=validate_gold(read_json(folder/"questions.json"),read_json(folder/"gold.json"))
    rows,masked=[],[]
    for key,item in inputs.items():
        pred=predictions[key];g=gold[item["question_id"]];response=pred.get("raw_response") or {}
        raw=response.get("response","");score=score_answer(raw,g["reference_answers"],g["answer_type"])
        relevant={e["passage_id"] for e in g["relevant_evidence"]}
        rank=next((r for r,pid in enumerate(item["seed_ids"],1) if pid in relevant),None)
        text_by_id={p["id"]:p["text"] for p in item["passages"]}
        quote_coverage=[]
        for slot in g["required_slots"]:
            es=[e for e in g["relevant_evidence"] if e["passage_id"] in slot["evidence_ids"]]
            quote_coverage.append(bool(es) and all(e["quote"] in text_by_id.get(e["passage_id"],"") for e in es))
        row={"key":key,"question_id":item["question_id"],"group":item["group"],
             "question":item["question"]["question"],"status":pred["status"],
             "reused_from":pred.get("reused_from"),"em":score["em"],"token_f1":score["token_f1"],
             "parse_ok":score["parse_ok"],"abstain":score["abstain"],
             "short_answer":score["short_answer"],"raw_answer":raw,
             "done_reason":response.get("done_reason"),"planned_input_tokens":item["input_tokens"],
             "measured_input_tokens":response.get("prompt_eval_count"),"output_tokens":response.get("eval_count"),
             "request_seconds":pred.get("request_seconds"),"extension_tokens":item["extension_tokens"],
             "extra_count":len(item["extra_ids"]),"seed_ids":";".join(item["seed_ids"]),
             "extra_ids":";".join(item["extra_ids"]),"annotated_hit_at_3":int(rank is not None),
             "annotated_rr_at_3":1/rank if rank else 0,
             "annotated_slot_quotes_present":sum(quote_coverage),"required_slot_count":len(quote_coverage)}
        if pred["status"]!="ok": row["em"]=row["token_f1"]=0.0
        rows.append(row)
        masked.append({"key":key,"question_id":item["question_id"],"question":row["question"],
                       "raw_answer":raw,"parsed":score["parsed"],
                       **{field:g.get(field,[]) for field in
                          ["reference_answers","required_slots","allowed_extras","critical_errors","relevant_evidence"]}})
    def group_summary(rs):
        result={}
        for group in GROUPS:
            ss=[r for r in rs if r["group"]==group]
            result[group]={"n":len(ss),"em":mean(r["em"] for r in ss),
                "token_f1":mean(r["token_f1"] for r in ss),
                "annotated_hit_at_3":mean(r["annotated_hit_at_3"] for r in ss),
                "annotated_mrr_at_3":mean(r["annotated_rr_at_3"] for r in ss),
                "annotated_slot_quote_coverage":sum(r["annotated_slot_quotes_present"] for r in ss)/sum(r["required_slot_count"] for r in ss),
                "mean_planned_input_tokens":mean(r["planned_input_tokens"] for r in ss),
                "mean_extra_passages":mean(r["extra_count"] for r in ss),
                "format_failures":sum(not r["parse_ok"] for r in ss),
                "non_ok":sum(r["status"]!="ok" for r in ss),
                "truncated":sum(r["done_reason"]=="length" for r in ss),
                "reused_positions":sum(bool(r["reused_from"]) for r in ss)}
        return result
    def pairs(rs):
        indexed={r["key"]:r for r in rs};out={}
        for left,right in [("H","D"),("X","H")]:
            name=left+"_minus_"+right
            out[name]={"n":len(gold),"identical_prompts":sum(inputs[q+":"+left]["prompt_identity"]==inputs[q+":"+right]["prompt_identity"] for q in gold)}
            for metric in ["em","token_f1"]:
                diffs=[indexed[q+":"+left][metric]-indexed[q+":"+right][metric] for q in gold]
                out[name][metric]={"mean_delta":mean(diffs),"improved":sum(d>1e-12 for d in diffs),
                    "worsened":sum(d< -1e-12 for d in diffs),"tied":sum(abs(d)<=1e-12 for d in diffs)}
        return out
    write_csv(folder/"automatic_scores.csv",rows)
    random.Random(20261028).shuffle(masked)
    keymap={}
    for n,r in enumerate(masked,1):
        rid=f"R{n:03}";keymap[rid]=r.pop("key");r["review_id"]=rid
    write_json(folder/"masked_review.json",masked);write_json(folder/"review_key.json",keymap)
    attempts=read_jsonl(folder/"attempts.jsonl")
    summary={"split":m["split"],"manifest_sha256":sha256_file(folder/"manifest.json"),
        "groups":group_summary(rows),"paired":pairs(rows),"planned_positions":len(rows),
        "unique_prompt_conditions":m["unique_prompt_conditions"],"actual_attempts":len(attempts),
        "reused_positions":sum(bool(r["reused_from"]) for r in rows),
        "actual_completed_attempts":sum(a.get("classified_status")=="ok" for a in attempts),
        "actual_request_seconds_sum":sum(a["request_seconds"] for a in attempts),
        "human_review_completed":False,
        "coverage_note":"Presence of frozen annotated source quotes, not exhaustive semantic evidence recall.",
        "latency_note":"Shared identical responses are not independent timing measurements; no per-method speed claim."}
    write_json(folder/"automatic_summary.json",summary)
    if (folder/"slot_review.json").exists():
        reviews=read_json(folder/"slot_review.json")
        if len(reviews)!=len(keymap) or {r["review_id"] for r in reviews}!=set(keymap):
            raise ValueError("Reviews must cover all positions exactly once.")
        for r in reviews:
            qid=keymap[r["review_id"]].split(":")[0]
            assert r["tp"]+r["fn"]==len(gold[qid]["required_slots"])
            assert r["fp"]==len(r["fp_reasons"])
        reviewed=apply_review_overrides(rows,reviews,keymap)
        write_csv(folder/"reviewed_answer_scores.csv",reviewed)
        summary["reviewed_answer_groups"]=group_summary(reviewed)
        summary["reviewed_pairs"]=pairs(reviewed)
        summary["slot_reviews"]={g:aggregate_review([r for r in reviews if keymap[r["review_id"]].endswith(":"+g)]) for g in GROUPS}
        write_json(folder/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    for name in ["prepare","run","analyze"]:
        s=sub.add_parser(name);s.add_argument("--run-dir",required=True)
        if name=="prepare":s.add_argument("--split",choices=["development","regression","holdout"],required=True)
    a=p.parse_args();{"prepare":prepare,"run":run,"analyze":analyze}[a.command](a)
