import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="KPIs Pagamentos", layout="wide")

# ---------- Leitura de dados ----------
@st.cache_data
def load_kpis(path: str):
    df = pd.read_parquet(path)

    # Normaliza nomes possivelmente diferentes
    candidates_date = [c for c in df.columns if c.lower() in ("day","date","transaction_date")]
    date_col = candidates_date[0] if candidates_date else "day"
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Recalcula derivadas caso não existam
    if "TPV" not in df.columns and "amount_transacted" in df.columns:
        daily = df.groupby("date")[["amount_transacted","quantity_transactions"]].sum().reset_index()
        daily = daily.rename(columns={"amount_transacted":"TPV","quantity_transactions":"Transactions"})
        daily["Average_Ticket"] = daily["TPV"]/daily["Transactions"]
        df = daily

    if "Transactions" not in df.columns and "quantity_transactions" in df.columns:
        df["Transactions"] = df["quantity_transactions"]

    if "Average_Ticket" not in df.columns:
        df["Average_Ticket"] = df["TPV"]/df["Transactions"]

    # Semana básica (sem estatísticas ainda)
    if "weekday" not in df.columns:
        df["weekday"] = df["date"].dt.day_name()

    # NÃO calcular estatísticas/alertas aqui; isso será feito após os filtros
    # TPV_MA7: se já existir no arquivo, mantém; se não, adiciona globalmente (será recalculado após filtros também)
    if "TPV_MA7" not in df.columns:
        df["TPV_MA7"] = df["TPV"].rolling(7, min_periods=1).mean()

    return df

data_file = "kpis.parquet"
data_path = Path(f"/app/data/gold/{data_file}")
if not data_path.exists():
    st.error(f"Arquivo {data_path} não encontrado. Gere o kpis_gold.csv na Parte 3.")
    st.stop()

df = load_kpis(str(data_path))

# --- Filtro de entidade (PF, PJ, Ambos) ---
st.sidebar.subheader("Entidade")
entity_col = "entity" if "entity" in df.columns else None
if entity_col is None:
    st.warning("Coluna 'entity' não encontrada no dataset. O filtro PF/PJ foi ignorado.")
    entity_option = "Ambos"
else:
    options = ["Ambos"] + sorted(df["entity"].dropna().unique().tolist())
    entity_option = st.sidebar.radio("Selecione a entidade", options, index=0)

# Aplicar filtro por entidade antes de qualquer outro filtro
df_ent = df.copy()
if entity_col and entity_option in ("PF", "PJ"):
    df_ent = df_ent[df_ent[entity_col] == entity_option]

# ---------- Filtros de período ----------
if df_ent.empty:
    st.warning("Sem dados após o filtro de entidade.")
    st.stop()

date_min, date_max = df_ent["date"].min(), df_ent["date"].max()
start, end = st.sidebar.date_input("Período", value=(date_min, date_max))
mask = (df_ent["date"] >= pd.to_datetime(start)) & (df_ent["date"] <= pd.to_datetime(end))
dfp = df_ent.loc[mask].copy()

if dfp.empty:
    st.warning("Sem dados para o período selecionado.")
    st.stop()

# ---------- (Re)Cálculos pós-filtro ----------
# Weekday do subconjunto
dfp["weekday"] = dfp["date"].dt.day_name()

# Média móvel 7d por entidade dentro do subconjunto (se houver coluna 'entity')
if "entity" in dfp.columns:
    dfp = dfp.sort_values(["entity","date"])
    dfp["TPV_MA7"] = dfp.groupby("entity")["TPV"].transform(lambda s: s.rolling(7, min_periods=1).mean())
else:
    dfp = dfp.sort_values("date")
    dfp["TPV_MA7"] = dfp["TPV"].rolling(7, min_periods=1).mean()

# Estatísticas por dia da semana no subconjunto
wk_stats = (
    dfp.groupby("weekday")["TPV"]
       .agg(["mean","std"])
       .reset_index()
       .rename(columns={"mean":"TPV_weekday_mean","std":"TPV_weekday_std"})
)
dfp = dfp.merge(wk_stats, on="weekday", how="left")

# Alerta por weekday (±2σ)
def alerta_weekday_row(row):
    std = 0.0 if pd.isna(row["TPV_weekday_std"]) else row["TPV_weekday_std"]
    mean = row["TPV_weekday_mean"]
    low, high = mean - 2*std, mean + 2*std
    if row["TPV"] < low: return "queda"
    if row["TPV"] > high: return "alta"
    return "ok"

dfp["alerta_weekday"] = dfp.apply(alerta_weekday_row, axis=1)

# --------------------------------------------------------------
# KPIs, gráficos e a tabela de alertas
# --------------------------------------------------------------

# ---------- KPIs topo ----------
def pct(a, b):
    if b in (0, np.nan, None) or pd.isna(b): return np.nan
    return (a/b - 1.0) * 100

tpv_total = dfp["TPV"].sum()
tx_total = dfp["Transactions"].sum()
avg_ticket = tpv_total/tx_total if tx_total else np.nan

# comparações (último dia vs D-1, média 7d, média 28d)
last = dfp.iloc[-1] if len(dfp)>0 else None
prev = dfp.iloc[-2] if len(dfp)>1 else None
d1 = pct(last["TPV"], prev["TPV"]) if (last is not None and prev is not None) else np.nan
w_avg = dfp.tail(7)["TPV"].mean() if len(dfp)>=2 else np.nan
m_avg = dfp.tail(28)["TPV"].mean() if len(dfp)>=2 else np.nan
dw = pct(last["TPV"], w_avg) if (last is not None and not pd.isna(w_avg)) else np.nan
dm = pct(last["TPV"], m_avg) if (last is not None and not pd.isna(m_avg)) else np.nan

ent_label = entity_option if (entity_col and entity_option in ("PF","PJ")) else "PF + PJ"
st.title(f"📊 KPIs de Pagamentos — {ent_label}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("TPV (período)", f"R$ {tpv_total:,.0f}")
c2.metric("Transações", f"{tx_total:,.0f}")
c3.metric("Ticket Médio", f"R$ {avg_ticket:,.2f}")
c4.metric("Δ vs D-1", "-" if pd.isna(d1) else f"{d1:,.1f}%")
c5.metric("Δ vs média 7d", "-" if pd.isna(dw) else f"{dw:,.1f}%")

# ---------- Tendência ----------
st.subheader("Tendência diária de TPV")
fig_tpv = px.line(dfp, x="date", y="TPV", title="TPV diário")
fig_tpv.add_scatter(x=dfp["date"], y=dfp["TPV_MA7"], mode="lines", name="Média móvel 7d")
st.plotly_chart(fig_tpv, use_container_width=True)

# ---------- Tendências normalizadas (TPV x Ticket Médio) ----------
st.subheader("Tendências normalizadas (TPV x Ticket Médio)")
norm = dfp[["date","TPV","Average_Ticket"]].copy()
for col in ["TPV","Average_Ticket"]:
    v = norm[col].values.astype(float)
    mn, mx = np.nanmin(v), np.nanmax(v)
    norm[col+"_norm"] = (v - mn)/(mx - mn) if mx!=mn else 0.0
fig_norm = px.line(norm, x="date", y=["TPV_norm","Average_Ticket_norm"],
                   labels={"value":"Escala (0-1)", "variable":"Série"})
st.plotly_chart(fig_norm, use_container_width=True)

# ---------- Gráficos separados ----------
colA, colB = st.columns(2)
with colA:
    st.markdown("**TPV diário**")
    st.plotly_chart(px.line(dfp, x="date", y="TPV"), use_container_width=True)
with colB:
    st.markdown("**Ticket Médio diário**")
    st.plotly_chart(px.line(dfp, x="date", y="Average_Ticket"), use_container_width=True)

# ---------- Comparação por dia da semana ----------
st.subheader("Comportamento por Dia da Semana")
order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dfp["weekday"] = pd.Categorical(dfp["weekday"], ordered=True, categories=order)
wk_stats = dfp.groupby("weekday")["TPV"].agg(["count","mean","std"]).reset_index()
wk_stats["lower"] = wk_stats["mean"] - 2*wk_stats["std"].fillna(0)
wk_stats["upper"] = wk_stats["mean"] + 2*wk_stats["std"].fillna(0)

fig_wk = px.bar(wk_stats, x="weekday", y="mean",
                title="TPV médio por dia da semana", labels={"mean":"TPV médio"})
fig_wk.add_scatter(x=wk_stats["weekday"], y=wk_stats["lower"], mode="markers+lines", name="Faixa -2σ")
fig_wk.add_scatter(x=wk_stats["weekday"], y=wk_stats["upper"], mode="markers+lines", name="+2σ")
st.plotly_chart(fig_wk, use_container_width=True)

# ---------- Alertas do período ----------
st.subheader("Alertas (baseados no padrão do dia da semana)")
alerts = dfp[dfp["alerta_weekday"].isin(["queda","alta"])][
    ["date","weekday","TPV","TPV_weekday_mean","TPV_weekday_std","alerta_weekday"]
].sort_values("date")
st.dataframe(alerts.rename(columns={
    "date":"Data",
    "weekday":"Dia da Semana",
    "TPV":"TPV",
    "TPV_weekday_mean":"Média (dia)",
    "TPV_weekday_std":"Desvio Padrão (dia)",
    "alerta_weekday":"Alerta"
}), use_container_width=True)

# ---------- Insight automático simples (sem LLM) ----------
st.subheader("Resumo automático (regra simples)")
if len(dfp):
    last_row = dfp.iloc[-1]
    d1_text = "-" if pd.isna(d1) else f"{d1:,.1f}%"
    base = (f"No dia {last_row['date'].date()}, o TPV foi R$ {last_row['TPV']:,.0f} "
            f"({('+' if (not pd.isna(d1) and d1>=0) else '')}{d1_text} vs D-1). "
            f"Para {last_row['weekday']}, a média histórica é R$ {last_row['TPV_weekday_mean']:,.0f} "
            f"± {last_row['TPV_weekday_std']:,.0f}. Status: {last_row['alerta_weekday']}.")
    st.info(base)