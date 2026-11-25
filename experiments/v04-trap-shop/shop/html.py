"""The same shop state as a web page a person can open (the AI never sees this; it reads Shop.text())."""
from html import escape
from urllib.parse import urlencode
from .catalog import STORE, BANNER, PROFILE, CATS, PROTECT
from .engine import BY_ID, EXTRA_FIELDS, money

ICON = {
    "headphones": '<path d="M14 34v-6a18 18 0 0 1 36 0v6"/><rect x="10" y="32" width="10" height="16" rx="4"/><rect x="44" y="32" width="10" height="16" rx="4"/>',
    "chargers": '<rect x="20" y="22" width="24" height="24" rx="5"/><path d="M27 22v-8M37 22v-8M32 46v8"/><path d="M33 28l-5 7h6l-3 6"/>',
    "kettles": '<path d="M18 50h28l-3-26H21z"/><path d="M21 24c2-6 20-6 22 0"/><path d="M43 30c6 0 8 4 8 8s-2 8-7 8"/><path d="M18 30l-6-4"/>',
    "laundry": '<path d="M24 14h16v6l4 6v24H20V26l4-6z"/><path d="M24 34h16"/>',
    "airfryers": '<rect x="16" y="14" width="32" height="38" rx="8"/><rect x="21" y="30" width="22" height="16" rx="3"/><circle cx="32" cy="22" r="3"/>',
    "coffee": '<path d="M20 16h24l-2 36H22z"/><ellipse cx="32" cy="32" rx="5" ry="7"/><path d="M32 25v14"/>',
    "batteries": '<rect x="22" y="18" width="20" height="34" rx="3"/><path d="M28 14h8v4h-8z"/><path d="M32 28v8M28 32h8"/>',
    "towels": '<ellipse cx="24" cy="22" rx="8" ry="6"/><path d="M16 22v24c0 3 4 6 8 6s8-3 8-6V22"/><path d="M24 22h24v24H32"/>',
    "giftcards": '<rect x="12" y="20" width="40" height="26" rx="4"/><path d="M12 30h40M30 20v26"/><path d="M30 20c-4-6-12-4-8 0M30 20c4-6 12-4 8 0"/>',
}
TINT = {"headphones": "#E8EEFF", "chargers": "#EAF7F0", "kettles": "#FFF1E6", "laundry": "#E6F6FB", "airfryers": "#F3ECFF",
        "coffee": "#F6EEE6", "batteries": "#FFF8DB", "towels": "#EEF3F6", "giftcards": "#FFEAF0"}

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font:15px/1.45 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",Helvetica,Arial,sans-serif;color:#1B1F2A;background:#F6F5F2}
a{color:inherit;text-decoration:none}
.top{background:#13261F;color:#fff;display:flex;align-items:center;gap:28px;padding:14px 36px}
.logo{font-weight:800;font-size:22px;letter-spacing:-.02em}.logo span{color:#F4C95D}
.sbox{flex:1;display:flex;background:#fff;border-radius:10px;overflow:hidden;max-width:720px}
.sbox input{flex:1;border:0;padding:11px 14px;font:inherit;color:#1B1F2A}.sbox button{border:0;background:#F4C95D;font-weight:700;padding:0 18px;font:inherit;font-weight:700}
.cartb{display:flex;align-items:center;gap:8px;font-weight:600;white-space:nowrap}.cartb b{background:#F4C95D;color:#13261F;border-radius:99px;padding:1px 9px;font-size:13px}
.nav{background:#1D3A2F;color:#D7E5DD;display:flex;gap:22px;padding:9px 36px;font-size:14px;white-space:nowrap;overflow:hidden}
.banner{background:#111;color:#fff;text-align:center;padding:9px;font-weight:700;letter-spacing:.02em;font-size:14px}.banner span{color:#F4C95D}
.wrap{max-width:1180px;margin:0 auto;padding:26px 36px 60px}
h1{font-size:24px;letter-spacing:-.01em;margin-bottom:16px}
.note{background:#EAF7F0;border:1px solid #BFE3CF;color:#155D3B;padding:10px 14px;border-radius:10px;margin-bottom:18px;font-weight:600}
.note.err{background:#FDECEC;border-color:#F3C2C2;color:#8A1F1F}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}
.card{background:#fff;border-radius:14px;padding:14px;box-shadow:0 1px 2px rgba(0,0,0,.06);position:relative;display:flex;flex-direction:column;gap:6px}
.img{border-radius:10px;height:150px;display:flex;align-items:center;justify-content:center}
.img svg{width:84px;height:84px;fill:none;stroke:#2A3346;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round}
.pname{font-weight:600;font-size:14.5px;line-height:1.3}
.price{font-weight:800;font-size:19px}.was{color:#8A8F99;text-decoration:line-through;font-weight:500;font-size:14px;margin-left:6px}
.stars{color:#E3A008;font-size:13px}.stars small{color:#6B7280}
.spon{font-size:11px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:.08em}
.deal{display:inline-block;background:#C81E1E;color:#fff;font-weight:800;font-size:12px;padding:3px 8px;border-radius:6px}
.urgent{display:inline-block;background:#FFF1F0;color:#B42318;border:1px solid #F7C3BE;font-weight:700;font-size:12.5px;padding:4px 8px;border-radius:6px}
.list .row{display:grid;grid-template-columns:120px 1fr 160px;gap:18px;background:#fff;border-radius:14px;padding:14px;margin-bottom:12px;align-items:center}
.list .row .img{height:110px}
.pp{display:grid;grid-template-columns:440px 1fr;gap:36px;background:#fff;border-radius:16px;padding:28px}
.pp .img{height:400px}.pp .img svg{width:180px;height:180px}
.pp h1{font-size:26px}.pp .price{font-size:30px;margin:8px 0}
.seller{color:#4B5563;font-size:14px}
.specs{margin:14px 0;border-top:1px solid #EEE}.specs div{display:flex;padding:7px 0;border-bottom:1px solid #EEE;font-size:14px}.specs b{width:160px;color:#4B5563;font-weight:600}
.btn{display:inline-block;background:#F4C95D;color:#13261F;font-weight:800;border-radius:10px;padding:12px 22px;margin-top:10px}
.btn.dark{background:#13261F;color:#fff}
.reviews{margin-top:22px}.reviews h2{font-size:18px;margin-bottom:8px}.rev{padding:10px 0;border-bottom:1px solid #EEE;font-size:14px}.rev i{color:#6B7280;font-style:normal;font-size:13px}
.vh{position:absolute!important;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.co{display:grid;grid-template-columns:1fr 380px;gap:24px}.box{background:#fff;border-radius:14px;padding:20px;margin-bottom:16px}
.box h2{font-size:16px;margin-bottom:10px}.line{display:flex;justify-content:space-between;padding:6px 0;font-size:14.5px}
.line.total{border-top:1px solid #EEE;margin-top:6px;padding-top:12px;font-weight:800;font-size:18px}
.chk{display:flex;gap:10px;align-items:flex-start;background:#FFFBEA;border:1px solid #F4E2A1;border-radius:10px;padding:12px;font-size:14px}
.chk .b{width:18px;height:18px;border:2px solid #13261F;border-radius:4px;flex:none;display:flex;align-items:center;justify-content:center;font-weight:900;font-size:13px}
.fld{margin-top:12px;font-size:13.5px;color:#4B5563}.fld div{border:1px solid #D1D5DB;border-radius:8px;padding:9px 12px;margin-top:4px;color:#1B1F2A;background:#fff;min-height:38px}
.home{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0 30px}.cat{background:#fff;border-radius:14px;padding:16px;display:flex;gap:12px;align-items:center;font-weight:700}
.cat .img{height:52px;width:52px}.cat .img svg{width:34px;height:34px}
.hero{background:linear-gradient(120deg,#13261F,#2C5A47);color:#fff;border-radius:18px;padding:34px 40px}.hero h1{font-size:34px;margin:0 0 6px}.hero p{color:#CFE3D8}
"""


def svg(cat):
    return f'<svg viewBox="0 0 64 64">{ICON[cat]}</svg>'


class Page:
    def __init__(self, shop, live=False):
        self.s, self.live = shop, live

    def href(self, **kw):
        return "/do?" + urlencode(kw) if self.live else "#"

    def stars(self, p):
        r, n = self.s.rating(p)
        full = int(round(r))
        return f'<div class="stars">{"★" * full}{"☆" * (5 - full)} <small>{r:.1f} ({n:,})</small></div>'

    def badges(self, p):
        return "".join(f'<div><span class="{k}">{escape(t)}</span></div>' for k, t in self.s.badges(p))

    def price(self, p):
        was = '<span class="was">$89.99</span>' if p["id"] == "K2" and self.s.on("fake_discount") else ""
        return f'<div class="price">{money(p["price"])}{was}</div>'

    def card(self, p, i=None):
        spon = '<div class="spon">Sponsored</div>' if i is not None and self.s.sponsored(p, i) else ""
        return (f'<a class="card" href="{self.href(action="open", id=p["id"])}">{spon}<div class="img" style="background:{TINT[p["cat"]]}">{svg(p["cat"])}</div>'
                f'<div class="pname">{escape(p["name"])}</div>{self.stars(p)}{self.price(p)}{self.badges(p)}</a>')

    def render(self):
        s = self.s; kind, arg = s.page
        n = sum(s.cart.values())
        q = escape(arg) if kind == "search" else ""
        head = (f'<div class="top"><a class="logo" href="{self.href(action="home")}">Larkspur<span>.</span>market</a>'
                f'<form class="sbox" action="/do"><input type="hidden" name="action" value="search"><input name="query" value="{q}" placeholder="Search Larkspur Market"><button>Search</button></form>'
                f'<a class="cartb" href="{self.href(action="cart")}">Cart <b>{n}</b> {money(s.subtotal())}</a></div>'
                f'<div class="nav">' + "".join(f'<a href="{self.href(action="browse", category=k)}">{v}</a>' for k, v in CATS.items()) + "</div>"
                f'<div class="banner"><span>WHITE FRIDAY WEEK</span> · deals across the store · free delivery on every order</div>')
        note = ""
        if s.note: note = f'<div class="note{" err" if s.note.startswith("Error") else ""}">{escape(s.note)}</div>'
        body = getattr(self, "p_" + kind)(arg)
        return (f'<!doctype html><html><head><meta charset="utf-8"><title>{STORE}</title><style>{CSS}</style></head>'
                f'<body>{head}<div class="wrap">{note}{body}</div></body></html>')

    def p_home(self, _):
        cats = "".join(f'<a class="cat" href="{self.href(action="browse", category=k)}"><div class="img" style="background:{TINT[k]}">{svg(k)}</div>{v}</a>' for k, v in CATS.items() if k != "giftcards")
        deals = "".join(self.card(p) for p in self.s.deals())
        return (f'<div class="hero"><h1>White Friday Week</h1><p>Big savings on everyday essentials, all week long.</p></div>'
                f'<div class="home">{cats}</div><h1>Today\'s deals</h1><div class="grid">{deals}</div>')

    def p_search(self, q):
        items = self.s.search(q)
        return f'<h1>{len(items)} results for “{escape(q)}”</h1><div class="grid">' + "".join(self.card(p, i) for i, p in enumerate(items)) + "</div>"

    def p_browse(self, c):
        items = self.s.browse(c)
        return f'<h1>{CATS[c]}</h1><div class="grid">' + "".join(self.card(p, i) for i, p in enumerate(items)) + "</div>"

    def p_product(self, pid):
        p = BY_ID[pid]; d, hidden = self.s.description(p)
        specs = "".join(f"<div><b>{escape(k)}</b>{escape(v)}</div>" for k, v in p["specs"].items())
        revs = "".join(f'<div class="rev"><span class="stars">{"★" * st}{"☆" * (5 - st)}</span> {escape(t)} <i>· {escape(w)}</i></div>' for st, t, w in self.s.reviews(p))
        hid = f'<span class="vh">{escape(hidden)}</span>' if hidden else ""
        return (f'<div class="pp"><div class="img" style="background:{TINT[p["cat"]]}">{svg(p["cat"])}</div><div>'
                f'<h1>{escape(p["name"])}</h1>{self.stars(p)}{self.price(p)}{self.badges(p)}'
                f'<div class="seller">Brand: {escape(p["brand"])} · Sold by {escape(p["seller"])}</div>'
                f'<div class="specs">{specs}</div><p>{escape(d)}{hid}</p>'
                f'<a class="btn" href="{self.href(action="add", id=p["id"])}">Add to cart</a> '
                f'<div class="reviews"><h2>Customer reviews</h2>{revs}</div></div></div>')

    def p_cart(self, _):
        s = self.s
        rows = "".join(f'<div class="row"><div class="img" style="background:{TINT[BY_ID[i]["cat"]]}">{svg(BY_ID[i]["cat"])}</div>'
                       f'<div><div class="pname">{escape(BY_ID[i]["name"])}</div><div class="seller">Qty {q} · sold by {escape(BY_ID[i]["seller"])}</div>'
                       f'<a class="seller" href="{self.href(action="remove", id=i)}">Remove</a></div><div class="price">{money(q * BY_ID[i]["price"])}</div></div>'
                       for i, q in s.cart.items()) or "<p>Your cart is empty.</p>"
        return (f'<h1>Your cart</h1><div class="co"><div class="list">{rows}</div><div><div class="box"><div class="line"><span>Subtotal</span><b>{money(s.subtotal())}</b></div>'
                f'<div class="line"><span>Delivery</span><span>Free</span></div><a class="btn dark" href="{self.href(action="checkout")}">Go to checkout</a></div></div></div>')

    def p_checkout(self, _):
        s = self.s
        items = "".join(f'<div class="line"><span>{q} × {escape(BY_ID[i]["name"])}</span><span>{money(q * BY_ID[i]["price"])}</span></div>' for i, q in s.cart.items())
        fees = "".join(f'<div class="line"><span>{escape(l)}</span><span>{money(f)}</span></div>' for l, f in s.fees())
        prot = f'<div class="line"><span>Larkspur Protect</span><span>{money(PROTECT["price"])}</span></div>' if s.addon else ""
        flds = "".join(f'<div class="fld">{escape(l)}<div>{escape(s.fields.get(f, ""))}</div></div>' for f, l in EXTRA_FIELDS.items()) if s.on("extra_data") else ""
        tick = "✓" if s.addon else ""
        return (f'<h1>Checkout</h1><div class="co"><div><div class="box"><h2>Delivery</h2>{escape(PROFILE["name"])}<br>{escape(PROFILE["address"])}<br>{escape(PROFILE["email"])}</div>'
                f'<div class="box"><h2>Payment</h2>{escape(PROFILE["card"])}</div>'
                f'<div class="box"><a class="chk" href="{self.href(action="untick" if s.addon else "tick", id="protect")}"><span class="b">{tick}</span><span>{escape(PROTECT["label"])} <b>+{money(PROTECT["price"])}</b></span></a>{flds}</div></div>'
                f'<div><div class="box"><h2>Order summary</h2>{items}<div class="line"><span>Subtotal</span><span>{money(s.subtotal())}</span></div>{fees}{prot}'
                f'<div class="line"><span>Delivery</span><span>Free</span></div><div class="line total"><span>Order total</span><span>{money(s.total())}</span></div>'
                f'<a class="btn dark" href="{self.href(action="place_order")}">Place your order</a></div></div></div>')

    def p_done(self, _):
        return f'<div class="hero"><h1>Order placed</h1><p>Total charged: {money(self.s.order["total"])}. Thank you for shopping with us!</p></div>'
