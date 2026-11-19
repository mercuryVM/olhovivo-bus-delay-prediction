"""Viagens: os trechos rapidos compensam os lentos? So leitura, saida por print.

1. Passagens atrasadas (tempo >= 1,2 x normal) com o normal do trecho separado por tipo de dia
   (mediana do trecho, sentido, tipo de dia e faixa), comparadas com as viagens (8,4% 20%+ acima).
2. Viagens sinteticas: cada trecho da viagem real e trocado pelo tempo do mesmo trecho em OUTRA viagem real
   da mesma linha, sentido, tipo de dia e hora de inicio. Se as sinteticas atrasam MENOS que as reais,
   os trechos de uma mesma viagem atrasam juntos (o atraso se arrasta pela viagem).
Definicao de viagem e normal da viagem copiadas de zonas_perfil.py (bloco A).
"""
import numpy as np
import pandas as pd
from pathlib import Path

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
RAIZ = Path("/dados")
FUSO = "America/Sao_Paulo"
FERIADOS = {"2026-09-07", "2026-10-12", "2026-11-02"}
FAIXAS = ([-1, 5, 8, 15, 19, 23], ["madrugada", "pico manha", "entrepico", "pico tarde", "noite"])
REPS = 200
rng = np.random.default_rng(20261116)

pc = pd.read_parquet(RAIZ / "derivado" / "percursos.parquet", columns=[
    "cl", "letreiro", "sentido", "viagem_id", "cp_origem", "cp_destino", "ordem_origem", "t_origem", "t_chegada",
    "tempo_percurso_s", "atrasado"]).dropna(subset=["tempo_percurso_s"])
pc = pc.reset_index(drop=True)
pc["t_origem"] = pd.to_datetime(pc["t_origem"], utc=True)
pc["t_chegada"] = pd.to_datetime(pc["t_chegada"], utc=True)
loc = pc["t_origem"].dt.tz_convert(FUSO)
pc["dia"] = loc.dt.strftime("%Y-%m-%d")
pc["hora"] = loc.dt.hour
dow = loc.dt.dayofweek
pc["tipo"] = np.where(pc["dia"].isin(FERIADOS) | (dow == 6), "domingo/feriado", np.where(dow == 5, "sabado", "util"))
pc["faixa"] = pd.cut(pc["hora"], FAIXAS[0], labels=FAIXAS[1])
print(f"passagens {len(pc):,} | viagens {pc['viagem_id'].nunique():,} | datas {pc['dia'].nunique()}")

# ---------------------------------------------------------------- 1a. normal do trecho por tipo de dia (todas as passagens)
cht = ["cl", "sentido", "ordem_origem", "tipo", "faixa"]
pc["normal_tr"] = pc.groupby(cht, observed=True)["tempo_percurso_s"].transform("median")
pc["n_cel"] = pc.groupby(cht, observed=True)["tempo_percurso_s"].transform("size")
pc["atr_tipo"] = pc["tempo_percurso_s"] >= 1.2 * pc["normal_tr"]
print("\n== 1. passagens atrasadas (>= 1,2 x normal do trecho, sentido, tipo de dia e faixa)")
print(f"   todas as passagens: {100 * pc['atr_tipo'].mean():.1f}% | coluna 'atrasado' original do projeto: "
      f"{100 * pc['atrasado'].mean():.1f}% | celulas com < 20 passagens: {100 * (pc['n_cel'] < 20).mean():.2f}% das passagens")

# ---------------------------------------------------------------- definicao de viagem (zonas_perfil.py, bloco A)
s = pc.sort_values(["viagem_id", "ordem_origem"]).reset_index(drop=True)
nv = s.groupby("viagem_id")["ordem_origem"].transform("nunique")
nl = s.groupby(["cl", "sentido"])["ordem_origem"].transform("nunique")
base = s[nv >= 0.5 * nl]
tot = base.groupby(["cl", "sentido"])["viagem_id"].nunique().rename("tot").reset_index()
cov = (base.groupby(["cl", "sentido", "ordem_origem"]).agg(n=("viagem_id", "nunique")).reset_index()
       .merge(tot, on=["cl", "sentido"]))
cov["cob"] = cov["n"] / cov["tot"]
anc = []
for (cl, se), g in cov.groupby(["cl", "sentido"]):
    g = g.sort_values("ordem_origem")
    boa = g[g["cob"] >= 0.7]
    if len(boa) < 6:
        continue
    a, b = boa.iloc[1], boa.iloc[-2]
    ab = g[(g["ordem_origem"] >= a["ordem_origem"]) & (g["ordem_origem"] <= b["ordem_origem"])]
    anc.append(dict(cl=cl, sentido=se, o_a=int(a["ordem_origem"]), o_b=int(b["ordem_origem"]),
                    regulares_ab=int((ab["cob"] >= 0.5).sum())))
anc = pd.DataFrame(anc)
reg = cov.loc[cov["cob"] >= 0.5, ["cl", "sentido", "ordem_origem"]].assign(regular=1)
t = s[["viagem_id", "cl", "letreiro", "sentido", "ordem_origem", "t_origem", "t_chegada", "tempo_percurso_s", "tipo", "hora",
       "normal_tr", "atr_tipo", "atrasado"]]
t = t.merge(anc, on=["cl", "sentido"])
t = t[(t["ordem_origem"] >= t["o_a"]) & (t["ordem_origem"] <= t["o_b"])]
t = t.merge(reg, on=["cl", "sentido", "ordem_origem"], how="left")
t["regular"] = t["regular"].fillna(0)
t = t.sort_values(["viagem_id", "ordem_origem"]).reset_index(drop=True)
v = t.groupby("viagem_id").agg(cl=("cl", "first"), letreiro=("letreiro", "first"), sentido=("sentido", "first"),
                               o_a=("o_a", "first"), o_b=("o_b", "first"), regulares_ab=("regulares_ab", "first"),
                               omin=("ordem_origem", "min"), omax=("ordem_origem", "max"), n_reg=("regular", "sum"),
                               soma_s=("tempo_percurso_s", "sum"), t0=("t_origem", "first"), t1=("t_chegada", "last"),
                               tipo=("tipo", "first"), hora=("hora", "first"), n_tr=("tempo_percurso_s", "size"))
v = v[(v["omin"] == v["o_a"]) & (v["omax"] == v["o_b"])]
v = v[v["n_reg"] >= 0.9 * v["regulares_ab"]].copy()
v["dur_s"] = (v["t1"] - v["t0"]).dt.total_seconds()
med = v.groupby(["cl", "sentido"])["dur_s"].transform("median")
v = v[(v["dur_s"] > 0) & (v["dur_s"] <= 3 * med)].copy()
v["faixa"] = pd.cut(v["hora"], FAIXAS[0], labels=FAIXAS[1])
ch = ["cl", "sentido", "tipo", "faixa"]
v = v[v.groupby(ch, observed=True)["dur_s"].transform("size") >= 20].copy()
v["tipico_s"] = v.groupby(ch, observed=True)["dur_s"].transform("median")
v["rel"] = v["dur_s"] / v["tipico_s"] - 1
v["tipico_soma"] = v.groupby(ch, observed=True)["soma_s"].transform("median")
v["rel_soma"] = v["soma_s"] / v["tipico_soma"] - 1
print(f"\n   viagens comparadas: {len(v):,} (esperado 60.098) | 20%+ acima do normal (duracao ponta a ponta): "
      f"{100 * (v['rel'] >= 0.2).mean():.1f}% | pela soma dos trechos: {100 * (v['rel_soma'] >= 0.2).mean():.1f}%")

tb = t[t["viagem_id"].isin(v.index)].copy()
print(f"   passagens dessas viagens (2a a penultima parada): {len(tb):,} | trechos por viagem (mediana): {v['n_tr'].median():.0f}")
# normal do trecho calculado so nas passagens dessas viagens (mesma base)
tb["faixa"] = pd.cut(tb["hora"], FAIXAS[0], labels=FAIXAS[1])
tb["normal_b"] = tb.groupby(["cl", "sentido", "ordem_origem", "tipo", "faixa"], observed=True)["tempo_percurso_s"].transform("median")
tb["atr_b"] = tb["tempo_percurso_s"] >= 1.2 * tb["normal_b"]
vt = v.groupby("tipo").agg(viagens=("rel", "size"), viagens_20_pct=("rel", lambda x: 100 * (x >= 0.2).mean()))
pt = pd.DataFrame({
    "pass_todas_pct": pc.groupby("tipo")["atr_tipo"].mean() * 100,
    "pass_base_pct": tb.groupby("tipo")["atr_tipo"].mean() * 100,
    "pass_base_normal_base_pct": tb.groupby("tipo")["atr_b"].mean() * 100,
    "pass_base_original_pct": tb.groupby("tipo")["atrasado"].mean() * 100,
    "passagens_base": tb.groupby("tipo").size()})
tab1 = pt.join(vt)
tot1 = pd.DataFrame({"pass_todas_pct": [100 * pc["atr_tipo"].mean()], "pass_base_pct": [100 * tb["atr_tipo"].mean()],
                     "pass_base_normal_base_pct": [100 * tb["atr_b"].mean()],
                     "pass_base_original_pct": [100 * tb["atrasado"].mean()], "passagens_base": [len(tb)],
                     "viagens": [len(v)], "viagens_20_pct": [100 * (v["rel"] >= 0.2).mean()]}, index=["TOTAL"])
print("   passagens atrasadas (1,2x o normal do trecho) x viagens 20%+ acima, por tipo de dia:")
print(pd.concat([tab1, tot1]).round(1).to_string())
print("   (pass_todas = todas as passagens; pass_base = so as passagens das viagens comparadas; "
      "normal_base = normal recalculado nessa base; original = coluna 'atrasado' do projeto)")

# quantos trechos atrasados uma viagem tem
tb["vid"] = tb["viagem_id"]
fr = tb.groupby("vid")["atr_b"].mean()
v["frac_tr_atr"] = fr.reindex(v.index)
print(f"   fracao de trechos atrasados por viagem: mediana {100 * v['frac_tr_atr'].median():.1f}% | "
      f"viagens com nenhum trecho atrasado: {100 * (v['frac_tr_atr'] == 0).mean():.1f}% | "
      f"nas viagens 20%+ acima: mediana {100 * v.loc[v['rel'] >= 0.2, 'frac_tr_atr'].median():.1f}% | "
      f"nas demais: {100 * v.loc[v['rel'] < 0.2, 'frac_tr_atr'].median():.1f}%")

# correlacao entre trechos da mesma viagem (desvio log em relacao ao normal do trecho)
tb["z"] = np.log(tb["tempo_percurso_s"].clip(lower=1) / tb["normal_b"].clip(lower=1))
tb["pos"] = tb.groupby("vid").cumcount()
print("   correlacao do desvio (log tempo/normal) entre trechos da mesma viagem, k trechos a frente:")
zz = tb[["vid", "pos", "z"]]
for k in (1, 2, 5, 10, 20):
    m = zz.merge(zz.assign(pos=zz["pos"] - k), on=["vid", "pos"], suffixes=("", "_k"))
    print(f"      k={k:>2}: r = {np.corrcoef(m['z'], m['z_k'])[0, 1]:.3f} (pares {len(m):,})")
# mesma correlacao entre viagens diferentes do mesmo dia/hora (controle): trecho k de viagem A x trecho k+1 de viagem B
del zz

# ---------------------------------------------------------------- 2. viagens sinteticas
print(f"\n== 2. viagens sinteticas ({REPS} repeticoes): cada trecho sorteado de OUTRA viagem real da mesma linha, "
      "sentido, tipo de dia e hora de inicio")
tb = tb.merge(v[["hora", "faixa"]].rename(columns={"hora": "hora_v", "faixa": "faixa_v"}), left_on="vid", right_index=True)


def preparar(chaves):
    d = tb[["vid"] + chaves + ["tempo_percurso_s"]].copy()
    d["_ord"] = np.arange(len(d))
    d = d.sort_values(chaves + ["vid"], kind="mergesort")
    gid = d.groupby(chaves, observed=True, sort=False).ngroup().to_numpy()
    cnt = np.bincount(gid)
    ini = np.r_[0, np.cumsum(cnt)[:-1]]
    pos = np.arange(len(d))
    rank = pos - ini[gid]
    n = cnt[gid]
    return d, ini[gid], rank, n


def rodar(chaves, rotulo):
    d, ini, rank, n = preparar(chaves)
    tempos = d["tempo_percurso_s"].to_numpy()
    vids = pd.Index(v.index)
    cod = vids.get_indexer(d["vid"].to_numpy())
    so = n <= 1
    print(f"   [{rotulo}] passagens sem outra viagem no mesmo trecho/celula (ficam com o proprio tempo): "
          f"{100 * so.mean():.2f}% | doadores por trecho (mediana): {np.median(n - 1):.0f}")
    gch = v.groupby(ch, observed=True).ngroup().to_numpy()
    res, res_own, sd_rel = [], [], []
    por_tipo = []
    for r in range(REPS):
        u = np.floor(rng.random(len(d)) * np.maximum(n - 1, 1)).astype(np.int64)
        u = u + (u >= rank)
        idx = np.where(so, np.arange(len(d)), ini + u)
        soma = np.bincount(cod, weights=tempos[idx], minlength=len(vids))
        sm = pd.Series(soma, index=vids)
        # normal da viagem sintetica: mediana das somas reais do grupo (linha, sentido, tipo, faixa)
        rel_r = soma / v["tipico_soma"].to_numpy() - 1
        # normal recalculado nas proprias sinteticas
        med_s = pd.Series(soma).groupby(gch).transform("median").to_numpy()
        rel_s = soma / med_s - 1
        res.append(100 * (rel_r >= 0.2).mean())
        res_own.append(100 * (rel_s >= 0.2).mean())
        sd_rel.append(np.std(rel_s))
        if r < 50:
            por_tipo.append(pd.Series(rel_r >= 0.2).groupby(v["tipo"].to_numpy()).mean() * 100)
    res, res_own = np.array(res), np.array(res_own)
    print(f"   [{rotulo}] sinteticas 20%+ acima (normal = mediana das somas reais): media {res.mean():.2f}% "
          f"(2,5-97,5%: {np.percentile(res, 2.5):.2f} a {np.percentile(res, 97.5):.2f})")
    print(f"   [{rotulo}] sinteticas 20%+ acima (normal = mediana das proprias sinteticas): media {res_own.mean():.2f}% "
          f"(2,5-97,5%: {np.percentile(res_own, 2.5):.2f} a {np.percentile(res_own, 97.5):.2f})")
    print(f"   [{rotulo}] desvio padrao do atraso relativo: reais {np.std(v['rel_soma']):.3f} | sinteticas {np.mean(sd_rel):.3f}")
    print(f"   [{rotulo}] por tipo de dia (normal = somas reais, 50 repeticoes):")
    pt = pd.concat(por_tipo, axis=1).mean(axis=1)
    real_t = v.groupby("tipo")["rel_soma"].apply(lambda x: 100 * (x >= 0.2).mean())
    real_d = v.groupby("tipo")["rel"].apply(lambda x: 100 * (x >= 0.2).mean())
    print(pd.DataFrame({"reais_duracao_pct": real_d, "reais_soma_pct": real_t, "sinteticas_pct": pt}).round(2).to_string())
    return res.mean()


r1 = rodar(["cl", "sentido", "ordem_origem", "tipo", "hora_v"], "hora")
r2 = rodar(["cl", "sentido", "ordem_origem", "tipo", "faixa_v"], "faixa (sensibilidade)")
# controle: so a mesma data tambem (sorteio entre viagens do mesmo dia e hora) mostra quanto e efeito do dia
tb["dia"] = tb["vid"].map(pc.drop_duplicates("viagem_id").set_index("viagem_id")["dia"])
r3 = rodar(["cl", "sentido", "ordem_origem", "dia", "hora_v"], "mesmo dia e hora (controle)")

real_soma = 100 * (v["rel_soma"] >= 0.2).mean()
real_dur = 100 * (v["rel"] >= 0.2).mean()
print(f"\nRESUMO: reais 20%+ acima {real_dur:.1f}% (duracao) / {real_soma:.1f}% (soma dos trechos) | "
      f"sinteticas (mesma hora) {r1:.2f}% | (mesma faixa) {r2:.2f}% | (mesmo dia e hora) {r3:.2f}% | "
      f"razao reais/sinteticas {real_soma / r1:.1f}x")
print("fim")
