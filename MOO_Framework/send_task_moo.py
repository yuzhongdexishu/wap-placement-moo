from __future__ import annotations
import argparse, json, os
from pathlib import Path
import requests

PROJECT_ROOT=Path(__file__).resolve().parent
URL=os.getenv("PLANNING_AGENT_URL","http://127.0.0.1:7001/a2a")

# One PM task runs both stages with SP3 (23 dBm) and SC1. The Sionna surrogate scores
# exactly 600 receivers per stage and rejects layouts with missing links.
PM_TASK=(
    "Run the full PM experiment in one task: both construction stages, frame and enclosure, "
    "each executing the complete 8-tool chain. SP3 sensor profile (23 dBm); SC1 priority scenario, whose zones "
    "carry weight 2 while the remaining space carries weight 1. Search WAP count from N=1; require all 600 receivers to meet min_rsrp_dbm=-140, min_rsrq_db=-19.5, min_sinr_db=0; population=80, "
    "generations=35, minimum_wap_distance_m=4.0, max_depth=3, ga_samples_per_tx=1000000, "
    "los=true, specular_reflection=true, diffraction=true, refraction=true. LLM: deepseek."
)

def _llm_key_available():
    if os.getenv("DEEPSEEK_API_KEY","").strip(): return True
    try:
        cfg=json.loads((PROJECT_ROOT/"config"/"llm_config.json").read_text(encoding="utf-8"))
        return bool(str(cfg.get("api_key") or "").strip())
    except Exception: return False

def main():
    ap=argparse.ArgumentParser(description="PM (MOO framework) experiment sender: one task, frame + enclosure, SP3 (23 dBm), SC1 priority scenario (zones weight 2).")
    ap.add_argument("--timeout",type=int,default=21600); ap.add_argument("--url",default=URL); a=ap.parse_args()
    if not _llm_key_available():
        print("ERROR: no DeepSeek key found (DEEPSEEK_API_KEY env or config/llm_config.json api_key); the LLM is mandatory and the task will fail.",flush=True)
        return
    task=PM_TASK
    r=requests.post(a.url,json={"role":"user","content":{"type":"text","text":task}},timeout=a.timeout)
    body=r.json(); text=body.get("content",{}).get("text","")
    result=json.loads(text) if text else body
    out=PROJECT_ROOT/"result"/"last_agent_response.json"; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2)); print(f"Saved: {out}")
    if r.status_code>=400 or result.get("status")!="success": raise SystemExit(1)

if __name__=="__main__":main()
