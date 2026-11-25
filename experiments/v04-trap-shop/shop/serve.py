"""Browse the shop as a person: python -m shop.serve --seed 1 --cond trap  ->  http://localhost:8040"""
import argparse, http.server, mimetypes, os
from urllib.parse import urlparse, parse_qs
from .engine import Shop, make_trip
from .html import Page, STATIC

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--cond", default="trap", choices=["trap", "control"]); ap.add_argument("--port", type=int, default=8040)
a = ap.parse_args()
trip = make_trip(a.seed, a.cond); shop = Shop(trip)
print("list:", trip["needs"], "traps:", trip["traps"], "budget:", trip["budget"])


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/do":
            shop.act({k: v[0] for k, v in parse_qs(u.query).items()})
            self.send_response(303); self.send_header("Location", "/"); self.end_headers(); return
        if u.path.startswith("/static/"):
            f = os.path.realpath(os.path.join(STATIC, u.path[len("/static/"):]))
            if f.startswith(os.path.realpath(STATIC) + os.sep) and os.path.isfile(f):
                self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(f)[0] or "application/octet-stream"); self.end_headers()
                self.wfile.write(open(f, "rb").read()); return
            self.send_response(404); self.end_headers(); return
        body = Page(shop, live=True).render().encode()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)

    def log_message(self, *x): pass


http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
