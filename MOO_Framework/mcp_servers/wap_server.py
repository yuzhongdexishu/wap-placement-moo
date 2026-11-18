from __future__ import annotations
import asyncio
import functools
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tools.wap_layout_optimization.define_problem import define_optimization_problem
from tools.wap_layout_optimization.build_surrogate import build_sionna_surrogate
from tools.wap_layout_optimization.generate_candidates import generate_and_evaluate_candidate_layouts
from tools.wap_layout_optimization.optimize import optimize_wap_layout

HOST=os.getenv("WAP_MCP_HOST","127.0.0.1"); PORT=int(os.getenv("WAP_MCP_PORT","8102"))
TOOLS={
    "define_optimization_problem": define_optimization_problem,
    "build_sionna_surrogate": build_sionna_surrogate,
    "generate_and_evaluate_candidate_layouts": generate_and_evaluate_candidate_layouts,
    "optimize_wap_layout": optimize_wap_layout,
}

def _simple_server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): return
        def _send(self,obj,code=200):
            raw=json.dumps(obj,ensure_ascii=False,default=str).encode(); self.send_response(code); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(raw))); self.end_headers(); self.wfile.write(raw)
        def do_GET(self): self._send({"status":"ok","server":"WAP Placement Optimization MCP Server","tools":list(TOOLS)})
        def do_POST(self):
            try:
                n=int(self.headers.get("Content-Length","0")); payload=json.loads(self.rfile.read(n) or b"{}")
                self._send(TOOLS[payload["tool"]](**payload.get("args",{})))
            except Exception as exc: self._send({"status":"error","error":f"{type(exc).__name__}: {exc}"},500)
    ThreadingHTTPServer((HOST,PORT),Handler).serve_forever()

def main():
    try: from mcp.server.fastmcp import FastMCP
    except Exception: return _simple_server()
    mcp=FastMCP("WAP Placement Optimization MCP Server",host=HOST,port=PORT,streamable_http_path="/mcp",json_response=True)

    def _async_tool(fn):
        # Run the sync tool in a worker thread so long computations never block the
        # FastMCP event loop (session init, list_tools, and other clients stay responsive).
        @functools.wraps(fn)
        async def wrapper(**kwargs):
            return await asyncio.to_thread(fn, **kwargs)
        return wrapper

    for name, fn in TOOLS.items(): mcp.tool(name=name)(_async_tool(fn))
    mcp.run(transport=os.getenv("MCP_TRANSPORT","streamable-http"))

if __name__=="__main__": main()
