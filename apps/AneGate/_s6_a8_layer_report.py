import json, os, statistics as st
ex=json.load(open("_s6_fx_proxy_vs_device.json"))["results"]
rows=[]
for L in range(16):
    f=f"_s6_fx_a8mlp_L{L}.json"
    if not os.path.exists(f): continue
    r=json.load(open(f))["results"]
    out={"L":L}
    for name in ["natural","chat","sky"]:
        diffs=[]; flips=0
        for s in r[name]["steps"]:
            e=next(x for x in ex[name]["steps"] if x["k"]==s["k"])
            diffs.append(abs(s["gap"]-e["gap"])); flips+=int(s["got"]!=e["got"])
        out[name]=(round(st.mean(diffs),3), flips)
    out["fails"]={n: r[n]["fails"] for n in r if r[n]["fails"]}
    rows.append(out)
print("layer  natural(mean|dgap|, flips)  chat  sky   margin-clear fails")
for o in sorted(rows, key=lambda o: -(o["natural"][0]+o["chat"][0]+o["sky"][0])):
    print(f"L{o['L']:2d}  {o['natural']}  {o['chat']}  {o['sky']}  {o['fails']}")
