from __future__ import annotations
import os, socket, subprocess, sys, time
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent
PROCESSES=[]
SERVICES=[
    ("BIM MCP",[sys.executable,"-m","mcp_servers.bim_server"],8101),
    ("WAP MCP",[sys.executable,"-m","mcp_servers.wap_server"],8102),
    ("Execution Agent",[sys.executable,"-m","agents.execution_agent"],7002),
    ("Planning Agent",[sys.executable,"-m","agents.planning_agent"],7001),
]

def _wait(port:int,timeout:float=60):
    deadline=time.time()+timeout
    while time.time()<deadline:
        with socket.socket() as s:
            s.settimeout(.3)
            if s.connect_ex(("127.0.0.1",port))==0:return
        time.sleep(.2)
    raise TimeoutError(f"port {port} did not open")

def main():
    env=os.environ.copy(); env["PYTHONPATH"]=str(PROJECT_ROOT)+os.pathsep+env.get("PYTHONPATH",""); env["MCP_TRANSPORT"]="streamable-http"
    # httpx (MCP client, openai SDK) honors the Windows system proxy from the registry;
    # when a system proxy is active, localhost MCP/A2A traffic must bypass it or the proxy
    # answers 502 for 127.0.0.1 endpoints.
    env["NO_PROXY"]="127.0.0.1,localhost"; env["no_proxy"]="127.0.0.1,localhost"
    try:
        for name,cmd,port in SERVICES:
            print(f"Starting {name} on {port} ...",flush=True)
            p=subprocess.Popen(cmd,cwd=PROJECT_ROOT,env=env)
            PROCESSES.append(p); _wait(port)
        print("All services are running. Run python send_task_moo.py for the frame and enclosure PM chain.",flush=True)
        while True:
            for p in PROCESSES:
                if p.poll() is not None: raise RuntimeError(f"service exited with code {p.returncode}")
            time.sleep(1)
    except KeyboardInterrupt: pass
    finally:
        for p in reversed(PROCESSES):
            if p.poll() is None: p.terminate()
        for p in reversed(PROCESSES):
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill()

if __name__=="__main__":main()
