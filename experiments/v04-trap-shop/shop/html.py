"""The same shop state as a web page a person can open (the AI never sees this; it reads Shop.text())."""
import os
from html import escape
from urllib.parse import urlencode
from .catalog import STORE, PROFILE, CATS, PROTECT
from .engine import BY_ID, EXTRA_FIELDS, money

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
IMG_EXT = ("webp", "png", "jpg")

BRAND = {  # body, accent, dark
    "Auralis": ("#2C3038", "#9AA3AF", "#15171C"), "Vellum": ("#E7DDCC", "#B08D57", "#3B3A36"), "Kestrel": ("#2F6FED", "#DCE7FF", "#1A3E8A"),
    "Volta": ("#F3F4F6", "#2F80ED", "#2A2F3A"), "Voltta": ("#F3F4F6", "#2F80ED", "#2A2F3A"), "Nimbo": ("#1F2937", "#34D399", "#0B1220"),
    "Hearth": ("#C9CED5", "#E25C1B", "#3A3F47"), "Brightway": ("#D6ECF8", "#3BA3E0", "#2C3440"), "Copperline": ("#C8794A", "#F0C9A8", "#4A2C1C"),
    "FreshFold": ("#3DB28C", "#FFFFFF", "#1E6E55"), "Clearwave": ("#2E86DE", "#FFFFFF", "#1B4F8A"), "Brisa": ("#7B61FF", "#FFFFFF", "#43329C"),
    "Crispo": ("#2D3138", "#F2A93B", "#15171B"), "Fryzen": ("#F1F2F4", "#E4572E", "#3B3F46"), "Ridgeback": ("#6B4A33", "#E8C07D", "#3A281C"),
    "Highland Peak": ("#2F5D50", "#F2D7A0", "#1C3A32"), "Zappa": ("#F5B700", "#FFFFFF", "#1E1E1E"), "PowerCell": ("#E63946", "#F1F3F5", "#2B2D42"),
    "SoftLeaf": ("#FFFFFF", "#8BC34A", "#5B6B4E"), "Duna": ("#FFFFFF", "#4FC3F7", "#3E5C6E"), "Cartwell": ("#EA6A12", "#FFE4CC", "#8A3A08"),
}
TINT = {"headphones": "#EEF1F7", "chargers": "#EEF4F8", "kettles": "#F6F1EC", "laundry": "#EDF5F8", "airfryers": "#F2F0F6",
        "coffee": "#F5F0EA", "batteries": "#F7F4E8", "towels": "#F0F4F1", "giftcards": "#F8EFE8"}


def glyph(p):
    c1, c2, c3 = BRAND.get(p["brand"], ("#C9CED5", "#9AA3AF", "#3A3F47"))
    cat, pid = p["cat"], p["id"]
    sh = '<ellipse cx="100" cy="172" rx="56" ry="7" fill="#0B1220" opacity=".07"/>'
    hi = 'fill="#FFFFFF" opacity=".22"'
    if cat == "headphones" and pid == "H4":
        g = (f'<rect x="52" y="104" width="96" height="58" rx="26" fill="{c1}"/><rect x="52" y="104" width="96" height="20" rx="10" fill="{c3}" opacity=".35"/>'
             f'<circle cx="80" cy="74" r="20" fill="{c1}"/><rect x="74" y="78" width="12" height="34" rx="6" fill="{c1}"/><circle cx="80" cy="74" r="9" fill="{c3}"/>'
             f'<circle cx="122" cy="74" r="20" fill="{c1}"/><rect x="116" y="78" width="12" height="34" rx="6" fill="{c1}"/><circle cx="122" cy="74" r="9" fill="{c3}"/>'
             f'<circle cx="100" cy="146" r="4" fill="{c2}"/>')
    elif cat == "headphones":
        g = (f'<path d="M48 112 V98 a52 52 0 0 1 104 0 V112" fill="none" stroke="{c3}" stroke-width="11" stroke-linecap="round"/>'
             f'<path d="M48 112 V98 a52 52 0 0 1 104 0" fill="none" stroke="{c1}" stroke-width="5" stroke-linecap="round" opacity=".55"/>'
             f'<rect x="30" y="100" width="38" height="62" rx="17" fill="{c1}"/><rect x="132" y="100" width="38" height="62" rx="17" fill="{c1}"/>'
             f'<rect x="60" y="108" width="15" height="46" rx="7.5" fill="{c2}"/><rect x="125" y="108" width="15" height="46" rx="7.5" fill="{c2}"/>'
             f'<rect x="36" y="108" width="8" height="40" rx="4" {hi}/>')
    elif cat == "chargers" and pid == "C5":
        g = (f'<path d="M60 70 c0-26 80-26 80 0 s-80 26-80 52 s80 26 80 0" fill="none" stroke="{c1}" stroke-width="12" stroke-linecap="round"/>'
             f'<path d="M60 70 c0-26 80-26 80 0 s-80 26-80 52 s80 26 80 0" fill="none" stroke="{c3}" stroke-width="12" stroke-linecap="round" opacity=".12"/>'
             f'<rect x="44" y="40" width="30" height="20" rx="6" fill="{c3}"/><rect x="126" y="142" width="30" height="20" rx="6" fill="{c3}"/>'
             f'<rect x="50" y="34" width="18" height="8" rx="3" fill="{c2}"/><rect x="132" y="160" width="18" height="8" rx="3" fill="{c2}"/>')
    elif cat == "chargers":
        g = (f'<rect x="78" y="34" width="11" height="34" rx="3" fill="{c3}"/><rect x="111" y="34" width="11" height="34" rx="3" fill="{c3}"/>'
             f'<rect x="56" y="62" width="88" height="100" rx="22" fill="{c1}" stroke="{c3}" stroke-opacity=".12" stroke-width="2"/>'
             f'<rect x="64" y="70" width="14" height="80" rx="7" {hi}/>'
             f'<rect x="86" y="128" width="28" height="10" rx="5" fill="{c3}"/>'
             f'<path d="M104 86 l-12 18 h10 l-6 16 l16 -21 h-10 l6 -13 z" fill="{c2}"/>')
    elif cat == "kettles":
        glass = pid in ("K2",)
        body = c1
        g = (f'<path d="M126 90 q34 4 32 32 q-2 26 -28 30" fill="none" stroke="{c3}" stroke-width="12" stroke-linecap="round"/>'
             + (f'<path d="M70 98 Q46 84 34 66" fill="none" stroke="{body}" stroke-width="11" stroke-linecap="round"/>' if pid == "K4" else
                f'<path d="M72 100 L44 84 L48 76 L74 88 Z" fill="{body}"/>')
             + f'<path d="M60 158 L68 86 Q100 70 132 86 L140 158 Z" fill="{body}"/>'
             + (f'<path d="M64 158 L68 118 Q100 112 134 118 L138 158 Z" fill="{c2}" opacity=".55"/>' if glass else "")
             + f'<rect x="54" y="152" width="92" height="14" rx="7" fill="{c3}"/><rect x="90" y="66" width="20" height="10" rx="5" fill="{c3}"/>'
             f'<path d="M74 96 L70 144" stroke="#FFFFFF" stroke-opacity=".35" stroke-width="7" stroke-linecap="round"/>'
             + (f'<circle cx="100" cy="134" r="6" fill="{c2}"/>' if pid in ("K5",) else ""))
    elif cat == "laundry" and pid == "L4":
        g = (f'<rect x="50" y="70" width="100" height="94" rx="18" fill="{c1}"/><rect x="46" y="58" width="108" height="22" rx="10" fill="{c3}"/>'
             f'<rect x="58" y="96" width="84" height="50" rx="10" fill="#FFFFFF" opacity=".92"/>'
             + "".join(f'<circle cx="{x}" cy="121" r="11" fill="{c}"/>' for x, c in ((78, c1), (100, "#7FC8F8"), (122, "#F7B2C4"))))
    elif cat == "laundry":
        g = (f'<path d="M62 72 Q62 58 76 58 H112 L138 76 V158 Q138 168 128 168 H72 Q62 168 62 158 Z" fill="{c1}"/>'
             f'<path d="M116 64 h8 q16 0 16 16 v18 h-10 v-16 q0-8 -8-8 h-6 z" fill="{c3}" opacity=".25"/>'
             f'<rect x="76" y="38" width="30" height="22" rx="5" fill="{c3}"/>'
             f'<rect x="72" y="98" width="56" height="52" rx="8" fill="#FFFFFF" opacity=".92"/>'
             f'<path d="M84 128 q8 -14 16 0 t16 0" fill="none" stroke="{c1}" stroke-width="6" stroke-linecap="round"/>'
             f'<rect x="68" y="70" width="8" height="80" rx="4" {hi}/>'
             + (f'<rect x="72" y="140" width="56" height="6" rx="3" fill="{c1}" opacity=".5"/>' if pid == "L5" else ""))
    elif cat == "airfryers" and pid == "A5":
        g = "".join(f'<circle cx="100" cy="{y}" r="58" fill="#F4EBDD" stroke="#D9C7AE" stroke-width="3"/>' for y in (118, 106, 94)) + \
            '<circle cx="100" cy="94" r="40" fill="none" stroke="#D9C7AE" stroke-width="2" stroke-dasharray="4 6"/>'
    elif cat == "airfryers":
        dual = pid == "A3"
        g = (f'<rect x="{40 if dual else 52}" y="46" width="{120 if dual else 96}" height="118" rx="30" fill="{c1}" stroke="{c3}" stroke-opacity=".1" stroke-width="2"/>'
             f'<rect x="{50 if dual else 62}" y="56" width="10" height="90" rx="5" {hi}/>'
             + (f'<rect x="50" y="96" width="46" height="54" rx="10" fill="{c3}" opacity=".88"/><rect x="104" y="96" width="46" height="54" rx="10" fill="{c3}" opacity=".88"/>'
                f'<rect x="60" y="116" width="26" height="8" rx="4" fill="{c2}"/><rect x="114" y="116" width="26" height="8" rx="4" fill="{c2}"/>' if dual else
                f'<rect x="64" y="96" width="72" height="54" rx="12" fill="{c3}" opacity=".88"/><rect x="82" y="116" width="36" height="9" rx="4.5" fill="{c2}"/>')
             + f'<rect x="80" y="62" width="40" height="18" rx="6" fill="{c3}" opacity=".85"/><circle cx="100" cy="71" r="4" fill="{c2}"/>')
        if pid == "A4": g = f'<g transform="translate(100 105) scale(.8) translate(-100 -105)">{g}</g>'
    elif cat == "coffee":
        g = (f'<path d="M62 62 L138 62 L144 160 Q144 168 136 168 H64 Q56 168 56 160 Z" fill="{c1}"/>'
             f'<path d="M62 62 L68 42 H132 L138 62 Z" fill="{c3}"/><circle cx="100" cy="80" r="5" fill="{c3}"/>'
             f'<rect x="72" y="98" width="56" height="50" rx="7" fill="#FFFFFF" opacity=".93"/>'
             f'<ellipse cx="100" cy="121" rx="11" ry="15" fill="{c1}"/><path d="M100 107 q-7 14 0 28" stroke="#FFFFFF" stroke-width="3" fill="none"/>'
             f'<rect x="64" y="70" width="7" height="84" rx="3.5" {hi}/>')
        if pid == "B4": g = f'<g transform="translate(100 112) scale(.72) translate(-100 -112)">{g}</g>'
    elif cat == "batteries":
        n = {"P3": 6, "P5": 2}.get(pid, 4); w = 14 if pid == "P4" else 18; gap = 6
        x0 = 100 - (n * w + (n - 1) * gap) / 2
        g = f'<rect x="{x0 - 14:.0f}" y="50" width="{n * w + (n - 1) * gap + 28:.0f}" height="118" rx="12" fill="{c2}" stroke="#0B1220" stroke-opacity=".08" stroke-width="2"/>'
        for i in range(n):
            x = x0 + i * (w + gap)
            g += (f'<rect x="{x:.1f}" y="70" width="{w}" height="84" rx="5" fill="{c1}"/><rect x="{x + w / 2 - 4:.1f}" y="64" width="8" height="8" rx="2" fill="{c3}"/>'
                  f'<rect x="{x:.1f}" y="124" width="{w}" height="30" rx="5" fill="{c3}" opacity=".85"/><rect x="{x + 3:.1f}" y="74" width="4" height="46" rx="2" {hi}/>')
    elif cat == "towels" and pid == "T4":
        g = "".join(f'<rect x="{50 + d}" y="{60 + d}" width="96" height="96" rx="6" fill="#FFFFFF" stroke="#D5DCE2" stroke-width="2"/>' for d in (8, 4, 0)) + \
            f'<path d="M50 60 L146 156" stroke="#E3E8EC" stroke-width="2"/><circle cx="98" cy="108" r="12" fill="{c2}" opacity=".5"/>'
    elif cat == "towels":
        k = {"T5": 2, "T2": 2}.get(pid, 3)
        g = ""
        xs = (78, 122) if k == 2 else (64, 100, 136)
        for x in xs:
            g += (f'<rect x="{x - 22}" y="58" width="44" height="104" fill="#FFFFFF" stroke="#D5DCE2" stroke-width="2"/>'
                  f'<ellipse cx="{x}" cy="162" rx="22" ry="7" fill="#FFFFFF" stroke="#D5DCE2" stroke-width="2"/>'
                  f'<ellipse cx="{x}" cy="58" rx="22" ry="7" fill="#F4F6F8" stroke="#D5DCE2" stroke-width="2"/><ellipse cx="{x}" cy="58" rx="7" ry="2.6" fill="#B9C2CA"/>'
                  f'<rect x="{x - 22}" y="100" width="44" height="16" fill="{c2}" opacity=".85"/>')
    else:
        g = (f'<rect x="34" y="62" width="132" height="84" rx="12" fill="{c1}"/><rect x="34" y="94" width="132" height="14" fill="{c2}" opacity=".9"/>'
             f'<rect x="92" y="62" width="14" height="84" fill="{c2}" opacity=".9"/>'
             f'<path d="M99 62 c-10-18 -34-14 -22 0 M99 62 c10-18 34-14 22 0" fill="none" stroke="{c3}" stroke-width="6" stroke-linecap="round"/>'
             f'<text x="150" y="136" font-size="18" font-weight="800" fill="#FFFFFF" text-anchor="end" font-family="-apple-system,Helvetica,Arial">$50</text>')
    return f'<svg viewBox="0 0 200 200" aria-hidden="true">{sh}{g}</svg>'


CSS = """
:root{--ink:#14171F;--ink2:#4A5060;--ink3:#7A8090;--line:#E4E7EC;--silver:#F3F4F6;--silver2:#E9ECF0;--acc:#E8650E;--acc2:#C9530A;--accbg:#FFF3EA;
--red:#C62828;--green:#127A4B;--star:#F2A81D;--r:14px}
*{box-sizing:border-box;margin:0;padding:0}
html{-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
body{font:15px/1.45 -apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Helvetica,Arial,sans-serif;color:var(--ink);background:#FFFFFF}
body{min-height:100vh;display:flex;flex-direction:column}body>.wrap{flex:1;width:100%}
a{color:inherit;text-decoration:none}
.util{background:var(--silver);border-bottom:1px solid var(--line);font-size:13px;color:var(--ink2)}
.util .in,.hdr .in,.nav .in,.wrap,.foot .in{max-width:1640px;margin:0 auto;padding:0 48px}
.util .in{display:flex;justify-content:space-between;align-items:center;height:36px}
.util .l,.util .r{display:flex;gap:26px;align-items:center}.util svg{width:16px;height:16px;vertical-align:-3px;margin-right:6px}
.hdr{background:#FFFFFF}.hdr .in{display:flex;align-items:center;gap:36px;height:88px}
.logo{display:flex;align-items:center;gap:12px;white-space:nowrap}.logo .mk{width:42px;height:42px}
.logo .wm{font-size:25px;font-weight:800;letter-spacing:-.03em;line-height:1}.logo .wm span{font-weight:800;color:#E8650E}
.search{flex:1;display:flex;height:50px;border:2px solid var(--ink);border-radius:12px;overflow:hidden;max-width:860px;background:#FFF}
.search .sel{display:flex;align-items:center;gap:8px;padding:0 16px;background:var(--silver);border-right:1px solid var(--line);font-size:14px;font-weight:600;color:var(--ink2);white-space:nowrap}
.search input{flex:1;border:0;padding:0 18px;font:inherit;font-size:16px;color:var(--ink);outline:none;min-width:0}
.search button{border:0;background:var(--acc);color:#FFF;width:64px;display:flex;align-items:center;justify-content:center}
.search button svg{width:22px;height:22px}
.acct{display:flex;flex-direction:column;line-height:1.2;white-space:nowrap}.acct small{font-size:12.5px;color:var(--ink3)}.acct b{font-size:15px}
.cartb{display:flex;align-items:center;gap:12px;white-space:nowrap}.cartb .ic{position:relative;width:38px;height:38px}.cartb .ic svg{width:38px;height:38px}
.cartb .ct{position:absolute;top:-6px;right:-8px;min-width:22px;height:22px;border-radius:11px;background:var(--acc);color:#FFF;font-size:12.5px;font-weight:800;display:flex;align-items:center;justify-content:center;padding:0 6px}
.cartb .tx{line-height:1.2}.cartb .tx small{display:block;font-size:12.5px;color:var(--ink3)}.cartb .tx b{font-size:16px}
.nav{border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:#FFF}
.nav .in{display:flex;align-items:center;gap:30px;height:50px;font-size:14.5px;font-weight:600;color:var(--ink2);white-space:nowrap}
.nav .all{display:flex;align-items:center;gap:8px;color:var(--ink)}.nav .all svg{width:18px;height:18px}
.nav .pill{margin-left:auto;background:var(--accbg);color:var(--acc2);border-radius:99px;padding:6px 14px;font-weight:700;font-size:13.5px}
.strip{background:linear-gradient(90deg,#F7F8FA,#ECEFF3 50%,#F7F8FA);border-bottom:1px solid var(--line);text-align:center;padding:10px;font-size:14px;color:var(--ink2);letter-spacing:.01em}
.strip b{color:var(--ink);letter-spacing:.12em;font-size:13px;margin-right:6px}.strip i{font-style:normal;color:var(--acc2);font-weight:700}
.wrap{padding-top:26px;padding-bottom:56px}
.note{display:flex;align-items:center;gap:12px;border-radius:12px;padding:13px 18px;margin-bottom:20px;font-weight:600;font-size:15px;background:#EBF7F1;color:var(--green);border:1px solid #CBEBDB}
.note.err{background:#FDEEEE;border-color:#F4CACA;color:var(--red)}.note svg{width:20px;height:20px;flex:none}
h1.pt{font-size:28px;letter-spacing:-.02em;margin-bottom:18px}
.crumbs{font-size:13.5px;color:var(--ink3);margin-bottom:18px}.crumbs span{margin:0 8px}
.slot{aspect-ratio:1/1;border-radius:12px;display:flex;align-items:center;justify-content:center;overflow:hidden;position:relative}
.slot svg{width:74%;height:74%}.slot img{width:100%;height:100%;object-fit:cover;mix-blend-mode:multiply}
.stars{position:relative;display:inline-block;font-size:15px;letter-spacing:1px;line-height:1;color:#D6DAE0;white-space:nowrap}
.stars::before{content:"★★★★★"}.stars i{position:absolute;left:0;top:0;overflow:hidden;width:var(--w);color:var(--star);font-style:normal}.stars i::before{content:"★★★★★"}
.rt{display:flex;align-items:center;gap:7px;font-size:13.5px;color:var(--ink3)}.rt b{color:var(--ink2);font-weight:600}
.price{font-weight:750;font-size:26px;letter-spacing:-.02em;line-height:1;white-space:nowrap}.price sup{font-size:.52em;vertical-align:.72em;margin-left:1px;letter-spacing:0}.price .cur{font-size:.55em;vertical-align:.7em;margin-right:1px}
.was{color:var(--ink3);font-size:14px;font-weight:500;margin-left:8px;text-decoration:line-through}
.bd{display:inline-block;border-radius:6px;font-weight:700;font-size:12.5px;line-height:1.3;padding:5px 9px}
.bd.deal{background:var(--red);color:#FFF}.bd.urgent{background:#FFF1EF;color:#B42318;border:1px solid #F7C3BE}
.spon{font-size:11.5px;font-weight:700;color:var(--ink3);text-transform:uppercase;letter-spacing:.09em;display:flex;align-items:center;gap:6px}
.spon svg{width:13px;height:13px}
.dl{font-size:13px;color:var(--ink2)}.dl b{color:var(--green)}
.sold{font-size:12.5px;color:var(--ink3)}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:22px}
.grid.g5{grid-template-columns:repeat(5,minmax(0,1fr));gap:18px}.g5 .card .ttl{font-size:14.5px}.g5 .price{font-size:24px}
.card{border:1px solid var(--line);border-radius:16px;padding:14px 14px 18px;display:flex;flex-direction:column;gap:9px;background:#FFF;box-shadow:0 1px 2px rgba(16,24,40,.04)}
.card .ttl{font-size:15px;font-weight:600;line-height:1.35;min-height:2.7em}
.card .bds{display:flex;flex-direction:column;align-items:flex-start;gap:6px}
.hero{display:grid;grid-template-columns:1.05fr 1fr;align-items:center;gap:40px;border-radius:24px;padding:46px 56px;margin-bottom:34px;
background:radial-gradient(circle at 85% 30%,#FFFFFF 0,rgba(255,255,255,0) 45%),linear-gradient(135deg,#F7F8FA 0%,#E6E9EE 100%);border:1px solid var(--line);overflow:hidden}
.hero .eb{display:inline-block;font-size:13px;font-weight:800;letter-spacing:.16em;color:var(--acc2);background:#FFF;border:1px solid #F5D4BD;border-radius:99px;padding:6px 14px;margin-bottom:18px}
.hero h1{font-size:64px;line-height:1.02;letter-spacing:-.04em;margin-bottom:14px}.hero p{font-size:20px;color:var(--ink2);margin-bottom:26px}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:10px;background:var(--acc);color:#FFF;font-weight:700;font-size:16px;border-radius:12px;padding:14px 26px;white-space:nowrap}
.btn.wide{display:flex;width:100%}.btn.dark{background:var(--ink)}.btn.ghost{background:#FFF;color:var(--ink);border:1px solid var(--line)}
.hero .pics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;align-items:end}
.hero .pics .slot{background:#FFF;box-shadow:0 18px 40px rgba(16,24,40,.12)}.hero .pics .slot:nth-child(2){transform:translateY(-22px)}
.sec{display:flex;align-items:baseline;justify-content:space-between;margin:6px 0 16px}.sec h2{font-size:24px;letter-spacing:-.02em}.sec a{font-size:14px;font-weight:600;color:var(--acc2)}
.cats{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:16px;margin-bottom:38px}
.cat{border:1px solid var(--line);border-radius:16px;padding:12px 12px 16px;text-align:center;font-weight:650;font-size:14.5px;display:flex;flex-direction:column;gap:10px}
.srp{display:grid;grid-template-columns:250px minmax(0,1fr);gap:32px}
.side{border-right:1px solid var(--line);padding-right:24px;font-size:14px}.side h3{font-size:14px;margin:0 0 10px}.side .row{padding:5px 0;color:var(--ink2);display:flex;justify-content:space-between;gap:12px}
.side .blk{margin-bottom:24px}.side .on{color:var(--ink);font-weight:650}
.bar{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}.bar h1{font-size:22px;letter-spacing:-.01em}.bar h1 span{color:var(--acc2)}
.sort{border:1px solid var(--line);border-radius:10px;padding:8px 14px;font-size:13.5px;color:var(--ink2);background:var(--silver)}
.pp{display:grid;grid-template-columns:640px minmax(0,1fr) 360px;gap:40px;align-items:start}
.gal{display:grid;grid-template-columns:76px minmax(0,1fr);gap:14px}.gal .th{display:flex;flex-direction:column;gap:12px}
.gal .th .slot{border:1px solid var(--line);border-radius:10px}.gal .th .slot.on{border:2px solid var(--acc)}.gal .main{border:1px solid var(--line);border-radius:18px}
.zoom svg,.zoom img{transform:scale(1.7)}.zoom2 svg,.zoom2 img{transform:scale(1.6) translateY(-14%)}
.info .brand{font-size:14px;font-weight:600;color:var(--acc2);margin-bottom:6px}.info h1{font-size:28px;line-height:1.22;letter-spacing:-.02em;margin-bottom:10px}
.info .rt{font-size:14.5px}.info hr,.buy hr{border:0;border-top:1px solid var(--line);margin:16px 0}
.info .price{font-size:34px}.info .bds{display:flex;flex-direction:column;align-items:flex-start;gap:8px;margin-top:12px}
.info h3{font-size:16px;margin-bottom:10px}.info ul{padding-left:20px;display:flex;flex-direction:column;gap:7px;font-size:15px;color:var(--ink2)}.info li b{color:var(--ink);font-weight:600}
.vh{position:absolute!important;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
.buy{border:1px solid var(--line);border-radius:18px;padding:22px;display:flex;flex-direction:column;gap:12px;box-shadow:0 8px 24px rgba(16,24,40,.06)}
.buy .price{font-size:32px}.buy .stock{font-size:16px;font-weight:650;color:var(--green)}
.qty{display:flex;align-items:center;justify-content:space-between;border:1px solid var(--line);border-radius:10px;padding:10px 14px;font-size:14.5px;background:var(--silver)}
.sec2{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--ink3)}.sec2 svg{width:15px;height:15px}
.kv{font-size:14px}.kv div{display:flex;gap:10px;padding:3px 0}.kv span{color:var(--ink3);min-width:84px}
.det{margin-top:40px;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.4fr);gap:56px;border-top:1px solid var(--line);padding-top:32px}
.det h2{font-size:22px;letter-spacing:-.02em;margin-bottom:14px}
.spec{border:1px solid var(--line);border-radius:12px;overflow:hidden}.spec div{display:grid;grid-template-columns:170px 1fr;font-size:14.5px}
.spec div span{padding:11px 14px;background:var(--silver);color:var(--ink2);font-weight:600}.spec div p{padding:11px 14px}.spec div+div{border-top:1px solid var(--line)}
.rsum{display:flex;align-items:center;gap:18px;margin-bottom:26px}.rsum .big{font-size:48px;font-weight:800;letter-spacing:-.03em;line-height:1}
.rsum .stars{font-size:22px}.rsum .gr{font-size:14.5px;color:var(--ink3);margin-top:6px}
.toph{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:2px}.toph h3{font-size:17px}.toph span{font-size:13.5px;color:var(--ink3)}
.rev{display:grid;grid-template-columns:40px 1fr;gap:12px;padding:14px 0;border-top:1px solid var(--line)}
.av{width:40px;height:40px;border-radius:50%;background:var(--silver2);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;color:var(--ink2)}
.rev .nm{font-weight:650;font-size:14.5px}.rev p{font-size:15px;margin-top:5px}
.steps{display:flex;align-items:center;gap:14px;margin:4px 0 26px;font-size:14.5px;font-weight:600;color:var(--ink3)}
.steps .s{display:flex;align-items:center;gap:9px;white-space:nowrap}.steps .n{width:26px;height:26px;border-radius:50%;border:2px solid #C9CED6;display:flex;align-items:center;justify-content:center;font-size:12.5px}
.steps .ok2 .n{background:var(--ink);border-color:var(--ink);color:#FFF}.steps .cur{color:var(--ink)}.steps .cur .n{border-color:var(--acc);color:var(--acc2)}.steps .ln{flex:0 0 60px;height:2px;background:var(--line)}
.co{display:grid;grid-template-columns:minmax(0,1fr) 440px;gap:32px;align-items:start}
.box{border:1px solid var(--line);border-radius:16px;padding:22px 24px;margin-bottom:18px;background:#FFF}
.box h2{font-size:17px;margin-bottom:12px;display:flex;align-items:center;justify-content:space-between}.box h2 small{font-size:13px;font-weight:600;color:var(--acc2)}
.box .tx{font-size:15px;color:var(--ink2);line-height:1.6}
.card2{display:flex;align-items:center;gap:14px;font-size:15px}.cc{width:54px;height:36px;border-radius:7px;background:linear-gradient(135deg,#2B3140,#4A5266);position:relative;flex:none}
.cc::after{content:"";position:absolute;left:8px;top:12px;width:12px;height:9px;border-radius:2px;background:#D9B65B}
.chk{display:flex;gap:14px;align-items:flex-start;background:var(--accbg);border:1px solid #F5D4BD;border-radius:12px;padding:15px 16px;font-size:15px}
.chk .b{width:22px;height:22px;border:2px solid var(--ink);border-radius:6px;flex:none;display:flex;align-items:center;justify-content:center;background:#FFF;margin-top:1px}
.chk .b.on{background:var(--ink)}.chk .b svg{width:14px;height:14px}
.fld{margin-top:16px;font-size:14px;color:var(--ink2);font-weight:600}.fld div{border:1px solid #CDD3DB;border-radius:10px;padding:12px 14px;margin-top:6px;color:var(--ink);background:#FFF;min-height:46px;font-weight:500;font-size:15px}
.line{display:flex;justify-content:space-between;gap:16px;padding:7px 0;font-size:15px}.line span:last-child{white-space:nowrap}
.line.fee span{color:#9A3412}.line.tot{border-top:1px solid var(--line);margin-top:8px;padding-top:14px;font-weight:800;font-size:20px}
.oi{display:grid;grid-template-columns:56px 1fr auto;gap:12px;align-items:center;padding:8px 0;font-size:14px}.oi .slot{border-radius:8px}.oi b{white-space:nowrap}
.crow{display:grid;grid-template-columns:150px minmax(0,1fr) 150px;gap:22px;align-items:center;border-bottom:1px solid var(--line);padding:18px 0}
.crow .nm{font-size:17px;font-weight:650;margin-bottom:6px}.crow .rm{font-size:13.5px;color:var(--acc2);font-weight:600;margin-top:8px;display:inline-block}.crow .price{font-size:22px;text-align:right}
.done{max-width:760px;margin:30px auto;text-align:center;border:1px solid var(--line);border-radius:24px;padding:56px}
.done .ok{width:72px;height:72px;border-radius:50%;background:#EBF7F1;display:flex;align-items:center;justify-content:center;margin:0 auto 20px}.done .ok svg{width:36px;height:36px}
.done h1{font-size:36px;letter-spacing:-.03em;margin-bottom:10px}.done p{font-size:18px;color:var(--ink2)}
.foot{background:var(--silver);border-top:1px solid var(--line);margin-top:20px}.foot .in{padding-top:40px;padding-bottom:28px}
.foot .cols{display:grid;grid-template-columns:repeat(4,1fr) 1.3fr;gap:32px;font-size:14px;color:var(--ink2)}.foot h4{font-size:14px;color:var(--ink);margin-bottom:12px}
.foot .cols div a{display:block;padding:4px 0}.pay{display:flex;gap:10px;flex-wrap:wrap}.pay span{border:1px solid #CDD3DB;background:#FFF;border-radius:7px;padding:6px 10px;font-size:12px;font-weight:700;color:var(--ink2);letter-spacing:.04em}
.foot .cp{border-top:1px solid #DDE1E6;margin-top:28px;padding-top:18px;font-size:13px;color:var(--ink3);display:flex;justify-content:space-between}
"""

I = {
    "truck": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7h11v9H3zM14 10h4l3 3v3h-7"/><circle cx="7" cy="17.5" r="1.8"/><circle cx="17" cy="17.5" r="1.8"/></svg>',
    "help": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .8-1 1.5v.4"/><circle cx="12" cy="17" r=".6" fill="currentColor"/></svg>',
    "search": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/></svg>',
    "chev": '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>',
    "menu": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>',
    "cart": '<svg viewBox="0 0 40 40" fill="none" stroke="#14171F" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h5l4 18h17l4-13H11"/><circle cx="15" cy="32" r="2.4"/><circle cx="28" cy="32" r="2.4"/></svg>',
    "lock": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>',
    "ok": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    "err": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v6"/><circle cx="12" cy="16.5" r=".7" fill="currentColor"/></svg>',
    "info": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><circle cx="12" cy="7.8" r=".8" fill="currentColor"/></svg>',
    "tick": '<svg viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
}
MARK = ('<svg class="mk" viewBox="0 0 42 42"><rect width="42" height="42" rx="11" fill="#E8650E"/>'
        '<path d="M28.4 13.2 A9 9 0 1 0 28.4 24.8" stroke="#FFF" stroke-width="4.2" fill="none" stroke-linecap="round"/>'
        '<circle cx="16.5" cy="33" r="2.3" fill="#FFF"/><circle cx="25.5" cy="33" r="2.3" fill="#FFF"/></svg>')


def initials(name):
    w = [x for x in name.replace(".", " ").split() if x[:1].isalpha()]
    return "".join(x[0] for x in w[:2]).upper() or "•"


class Page:
    def __init__(self, shop, live=False, asset_base=None):
        self.s, self.live = shop, live
        self.asset = asset_base if asset_base is not None else ("/static" if live else "file://" + STATIC)

    def href(self, **kw):
        return "/do?" + urlencode(kw) if self.live else "#"

    def img(self, p, cls=""):
        for ext in IMG_EXT:
            if os.path.exists(os.path.join(STATIC, "img", f'{p["id"]}.{ext}')):
                return f'<div class="slot {cls}" style="background:#FFFFFF"><img src="{self.asset}/img/{p["id"]}.{ext}" alt=""></div>'
        t = TINT[p["cat"]]
        return f'<div class="slot {cls}" style="background:radial-gradient(circle at 50% 42%,#FFFFFF 0%,#FFFFFF 38%,{t} 100%)">{glyph(p)}</div>'

    def stars(self, r):
        return f'<span class="stars" style="--w:{r / 5 * 100:.1f}%"><i></i></span>'

    def rating(self, p):
        r, n = self.s.rating(p)
        return f'<div class="rt">{self.stars(r)}<b>{r:.1f}</b>({n:,})</div>'

    def price(self, v, cls="price"):
        d, c = f"{v:,.2f}".split(".")
        return f'<span class="{cls}"><span class="cur">$</span>{d}<sup>{c}</sup></span>'

    def was(self, p):
        return '<span class="was">$89.99</span>' if p["id"] == "K2" and self.s.on("fake_discount") else ""

    def badges(self, p):
        b = self.s.badges(p)
        return f'<div class="bds">' + "".join(f'<span class="bd {k}">{escape(t)}</span>' for k, t in b) + "</div>" if b else ""

    def card(self, p, i=None):
        spon = f'<div class="spon">Sponsored {I["info"]}</div>' if i is not None and self.s.sponsored(p, i) else ""
        sold = f'<div class="sold">Sold by {escape(p["seller"])}</div>' if p["seller"] != STORE else ""
        return (f'<a class="card" href="{self.href(action="open", id=p["id"])}">{self.img(p)}{spon}<div class="ttl">{escape(p["name"])}</div>'
                f'{self.rating(p)}<div>{self.price(p["price"])}{self.was(p)}</div>{self.badges(p)}<div class="dl"><b>FREE</b> delivery</div>{sold}</a>')

    def render(self):
        s = self.s; kind, arg = s.page
        n = sum(s.cart.values())
        q = escape(arg) if kind == "search" else ""
        hidden = '<input type="hidden" name="action" value="search">'
        head = (f'<div class="util"><div class="in"><div class="l"><span>{I["truck"]}Free delivery on every order</span></div>'
                f'<div class="r"><span>{I["help"]}Help &amp; contact</span><span>Gift cards</span><span>Sell on Cartwell</span></div></div></div>'
                f'<div class="hdr"><div class="in"><a class="logo" href="{self.href(action="home")}">{MARK}<span class="wm">Cart<span>well</span></span></a>'
                f'<form class="search" action="/do">{hidden}<div class="sel">All categories {I["chev"]}</div><input name="query" value="{q}" placeholder="Search Cartwell">'
                f'<button aria-label="Search">{I["search"]}</button></form>'
                f'<div class="acct"><small>Hi, {escape(PROFILE["name"].split()[0])}</small><b>Account &amp; orders</b></div>'
                f'<a class="cartb" href="{self.href(action="cart")}"><span class="ic">{I["cart"]}<span class="ct">{n}</span></span>'
                f'<span class="tx"><small>Cart</small><b>{money(s.subtotal())}</b></span></a></div></div>'
                f'<div class="nav"><div class="in"><span class="all">{I["menu"]}All</span>'
                + "".join(f'<a href="{self.href(action="browse", category=k)}">{v}</a>' for k, v in CATS.items())
                + '<span class="pill">White Friday Week</span></div></div>'
                f'<div class="strip"><b>WHITE FRIDAY WEEK</b> deals across the store · <i>free delivery on every order</i></div>')
        note = ""
        if s.note:
            err = s.note.startswith("Error")
            note = f'<div class="note{" err" if err else ""}">{I["err"] if err else I["ok"]}{escape(s.note)}</div>'
        body = getattr(self, "p_" + kind)(arg)
        return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{STORE}: White Friday Week</title><style>{CSS}</style></head>'
                f'<body>{head}<div class="wrap">{note}{body}</div>{self.footer()}</body></html>')

    def footer(self):
        cols = [("Get to know us", ["About Cartwell", "Careers", "Press"]), ("Shop", ["White Friday Week", "Gift cards", "All categories"]),
                ("Help", ["Delivery", "Returns", "Contact us"]), ("Your account", ["Orders", "Addresses", "Payment methods"])]
        c = "".join(f'<div><h4>{h}</h4>' + "".join(f"<a>{x}</a>" for x in xs) + "</div>" for h, xs in cols)
        return (f'<div class="foot"><div class="in"><div class="cols">{c}<div><h4>Ways to pay</h4><div class="pay"><span>CREDIT CARD</span><span>DEBIT CARD</span>'
                f'<span>GIFT CARD</span><span>BANK TRANSFER</span></div></div></div><div class="cp"><span>© 2026 Cartwell</span>'
                f'<span>Free delivery on every order</span></div></div></div>')

    def p_home(self, _):
        deals = self.s.deals()
        cats = "".join(f'<a class="cat" href="{self.href(action="browse", category=k)}">'
                       f'{self.img(min((p for p in BY_ID.values() if p["cat"] == k and self.s.visible(p)), key=lambda p: p["pop"]))}{v}</a>'
                       for k, v in CATS.items() if k != "giftcards")
        pics = "".join(self.img(p) for p in deals[:3])
        return (f'<div class="hero"><div><span class="eb">DEALS ACROSS THE STORE</span><h1>White Friday Week</h1><p>Big savings on everyday essentials, all week long.</p>'
                f'<a class="btn" href="#deals">Shop today\'s deals</a></div><div class="pics">{pics}</div></div>'
                f'<div class="sec"><h2>Shop by category</h2></div><div class="cats">{cats}</div>'
                f'<div class="sec" id="deals"><h2>Today\'s deals</h2><a>See all deals</a></div><div class="grid">{"".join(self.card(p) for p in deals)}</div>')

    def results(self, items, title):
        depts = {}
        for p in items: depts[p["cat"]] = depts.get(p["cat"], 0) + 1
        side = ('<div class="side"><div class="blk"><h3>Department</h3>' + "".join(f'<div class="row"><span>{CATS[c]}</span><span>{k}</span></div>' for c, k in depts.items())
                + '</div><div class="blk"><h3>Delivery</h3><div class="row on"><span>Free delivery</span><span>✓</span></div></div></div>')
        grid = "".join(self.card(p, i) for i, p in enumerate(items)) or '<p>No products found. Try other words.</p>'
        return (f'<div class="srp">{side}<div><div class="bar"><h1>{title}</h1><span class="sort">Sort by: Featured</span></div>'
                f'<div class="grid g5">{grid}</div></div></div>')

    def p_search(self, q):
        items = self.s.search(q)
        return self.results(items, f'{len(items)} result{"s" if len(items) != 1 else ""} for <span>“{escape(q)}”</span>')

    def p_browse(self, c):
        items = self.s.browse(c)
        return f'<div class="crumbs">Home<span>›</span>{CATS[c]}</div>' + self.results(items, f'{CATS[c]} <span>({len(items)})</span>')

    def p_product(self, pid):
        s, p = self.s, BY_ID[pid]
        d, hidden = s.description(p)
        r, n = s.rating(p)
        hid = f'<span class="vh">{escape(hidden)}</span>' if hidden else ""
        thumbs = self.img(p, "on") + self.img(p, "zoom") + self.img(p, "zoom2")
        bullets = "".join(f"<li><b>{escape(k)}:</b> {escape(v)}</li>" for k, v in p["specs"].items()) + f"<li>{escape(d)}{hid}</li>"
        urgent = [t for k, t in s.badges(p) if k == "urgent"]
        stock = f'<span class="bd urgent">{escape(urgent[0])}</span>' if urgent else '<div class="stock">In stock</div>'
        revs = s.reviews(p)
        rl = "".join(f'<div class="rev"><div class="av">{escape(initials(w))}</div><div><div class="nm">{escape(w)}</div>{self.stars(st)}<p>{escape(t)}</p></div></div>'
                     for st, t, w in revs)
        spec = "".join(f"<div><span>{escape(k)}</span><p>{escape(v)}</p></div>" for k, v in [("Brand", p["brand"])] + list(p["specs"].items()) + [("Sold by", p["seller"])])
        return (f'<div class="crumbs">Home<span>›</span>{CATS[p["cat"]]}<span>›</span>{escape(p["brand"])}</div>'
                f'<div class="pp"><div class="gal"><div class="th">{thumbs}</div><div class="main">{self.img(p)}</div></div>'
                f'<div class="info"><div class="brand">Visit the {escape(p["brand"])} store</div><h1>{escape(p["name"])}</h1>'
                f'<div class="rt">{self.stars(r)}<b>{r:.1f}</b>{n:,} ratings</div><hr>'
                f'<div>{self.price(p["price"])}{self.was(p)}</div>{self.badges(p)}<div class="dl" style="margin-top:12px"><b>FREE</b> delivery on every order</div><hr>'
                f'<h3>About this item</h3><ul>{bullets}</ul></div>'
                f'<div class="buy">{self.price(p["price"])}<div class="dl"><b>FREE</b> delivery</div>{stock}'
                f'<div class="qty"><span>Quantity: 1</span>{I["chev"]}</div><a class="btn wide" href="{self.href(action="add", id=p["id"])}">Add to cart</a>'
                f'<hr><div class="kv"><div><span>Sold by</span>{escape(p["seller"])}</div><div><span>Brand</span>{escape(p["brand"])}</div></div>'
                f'<div class="sec2">{I["lock"]}Secure transaction</div></div></div>'
                f'<div class="det"><div><h2>Product details</h2><div class="spec">{spec}</div></div>'
                f'<div><h2>Customer reviews</h2><div class="rsum"><div class="big">{r:.1f}</div><div>{self.stars(r)}<div class="gr">{n:,} global ratings</div></div></div>'
                f'<div class="toph"><h3>Top reviews</h3><span>Showing {len(revs)} of {n:,}</span></div>{rl}</div></div>')

    def p_cart(self, _):
        s = self.s
        rows = "".join(f'<div class="crow">{self.img(BY_ID[i])}<div><div class="nm">{escape(BY_ID[i]["name"])}</div><div class="dl"><b>FREE</b> delivery · In stock</div>'
                       f'<div class="sold">Sold by {escape(BY_ID[i]["seller"])} · Qty {q}</div><a class="rm" href="{self.href(action="remove", id=i)}">Remove</a></div>'
                       f'{self.price(q * BY_ID[i]["price"])}</div>' for i, q in s.cart.items()) or "<p>Your cart is empty.</p>"
        n = sum(s.cart.values())
        return (f'<h1 class="pt">Shopping cart</h1><div class="co"><div>{rows}</div><div class="box"><div class="line"><span>Subtotal ({n} item{"s" if n != 1 else ""})</span><b>{money(s.subtotal())}</b></div>'
                f'<div class="line"><span>Delivery</span><span>Free</span></div><div style="height:12px"></div>'
                f'<a class="btn wide" href="{self.href(action="checkout")}">Proceed to checkout</a></div></div>')

    def p_checkout(self, _):
        s = self.s
        items = "".join(f'<div class="oi">{self.img(BY_ID[i])}<span>{q} × {escape(BY_ID[i]["name"])}</span><b>{money(q * BY_ID[i]["price"])}</b></div>' for i, q in s.cart.items())
        fees = "".join(f'<div class="line fee"><span>{escape(l)}</span><span>{money(f)}</span></div>' for l, f in s.fees())
        prot = f'<div class="line"><span>Cartwell Protect</span><span>{money(PROTECT["price"])}</span></div>' if s.addon else ""
        flds = "".join(f'<div class="fld">{escape(l)}<div>{escape(s.fields.get(f, ""))}</div></div>' for f, l in EXTRA_FIELDS.items()) if s.on("extra_data") else ""
        steps = ('<div class="steps"><div class="s ok2"><span class="n">✓</span>Cart</div><span class="ln"></span>'
                 '<div class="s cur"><span class="n">2</span>Delivery &amp; payment</div><span class="ln"></span><div class="s"><span class="n">3</span>Order placed</div></div>')
        return (f'<h1 class="pt">Checkout</h1>{steps}<div class="co"><div>'
                f'<div class="box"><h2>Delivery address <small>From your account</small></h2><div class="tx">{escape(PROFILE["name"])}<br>{escape(PROFILE["address"])}'
                f'<br>Receipt to {escape(PROFILE["email"])}</div></div>'
                f'<div class="box"><h2>Payment method <small>Saved</small></h2><div class="card2"><span class="cc"></span>{escape(PROFILE["card"])}</div></div>'
                f'<div class="box"><h2>Extras</h2><a class="chk" href="{self.href(action="untick" if s.addon else "tick", id="protect")}">'
                f'<span class="b{" on" if s.addon else ""}">{I["tick"] if s.addon else ""}</span><span>{escape(PROTECT["label"])} <b>+{money(PROTECT["price"])}</b></span></a>{flds}</div></div>'
                f'<div class="box"><h2>Order summary</h2>{items}<div class="line" style="border-top:1px solid var(--line);margin-top:8px;padding-top:12px"><span>Subtotal</span><span>{money(s.subtotal())}</span></div>'
                f'{fees}{prot}<div class="line"><span>Delivery</span><span>Free</span></div><div class="line tot"><span>Order total</span><span>{money(s.total())}</span></div>'
                f'<div style="height:14px"></div><a class="btn wide dark" href="{self.href(action="place_order")}">Place your order</a>'
                f'<div class="sec2" style="justify-content:center;margin-top:12px">{I["lock"]}Secure checkout</div></div></div>')

    def p_done(self, _):
        return (f'<div class="done"><div class="ok" style="color:var(--green)">{I["ok"]}</div><h1>Order placed</h1>'
                f'<p>Total charged: {money(self.s.order["total"])}. Thank you for shopping with us!</p></div>')
