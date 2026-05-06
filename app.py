# ==========================================
# app.py
# Sistema de análise de carteira B3
# Streamlit Cloud Ready
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="Carteira B3",
    layout="wide"
)

ARQUIVO_CARTEIRA = "carteira.csv"

# ==========================================
# CACHE
# ==========================================

def limpar_ticker(ticker):
    ticker = (
        str(ticker)
        .strip()
        .replace("\u00A0", "")
        .replace(" ", "")
        .upper()
    )

    if ticker.endswith(".SA"):
        ticker = ticker.replace(".SA", "")

    return ticker

@st.cache_data(ttl=3600)
def baixar_historico(ticker, periodo="1y"):

    ticker_limpo = limpar_ticker(ticker)

    ticker_yf = f"{ticker_limpo}.SA"

    ativo = yf.Ticker(ticker_yf)

    historico = ativo.history(period=periodo)

    if historico.empty:
        raise Exception(
            f"Sem dados para {ticker_yf}"
        )

    dividendos = ativo.dividends

    return historico, dividendos
    
# ==========================================
# INDICADORES
# ==========================================

def calcular_indicadores(df, dividendos):

    mensal = df.resample("M").last()

    mensal["Retorno"] = mensal["Close"].pct_change()

    # MM
    mensal["MM9"] = SMAIndicator(
        mensal["Close"],
        window=9
    ).sma_indicator()

    mensal["MM21"] = SMAIndicator(
        mensal["Close"],
        window=21
    ).sma_indicator()

    # RSI
    mensal["RSI"] = RSIIndicator(
        mensal["Close"],
        window=14
    ).rsi()

    # Drawdown
    rolling_max = mensal["Close"].cummax()

    mensal["Drawdown"] = (
        mensal["Close"] - rolling_max
    ) / rolling_max

    # Momentum
    mensal["Momentum"] = (
        mensal["Close"].pct_change(6)
    )

    # Dividendos
    dividendos = dividendos.resample("M").sum()

    mensal["Dividendos"] = dividendos

    mensal["Dividendos"] = (
        mensal["Dividendos"]
        .fillna(0)
    )

    mensal["Dividendos_12M"] = (
        mensal["Dividendos"]
        .rolling(12)
        .sum()
    )

    mensal["DY"] = (
        mensal["Dividendos_12M"]
        / mensal["Close"]
    ) * 100

    return mensal

# ==========================================
# SCORE
# ==========================================

def calcular_score(df):

    ultimo = df.iloc[-1]

    score_quant = 0

    # Tendência
    if ultimo["MM9"] > ultimo["MM21"]:
        score_quant += 30

    # RSI
    if 40 <= ultimo["RSI"] <= 70:
        score_quant += 20

    # Momentum
    if ultimo["Momentum"] > 0:
        score_quant += 20

    # Drawdown
    if ultimo["Drawdown"] > -0.20:
        score_quant += 20

    # Dividend Yield
    score_fund = 0

    if ultimo["DY"] >= 10:
        score_fund += 40

    elif ultimo["DY"] >= 6:
        score_fund += 30

    elif ultimo["DY"] >= 4:
        score_fund += 20

    # Dividendos consistentes
    if ultimo["Dividendos_12M"] > 0:
        score_fund += 30

    # Drawdown saudável
    if ultimo["Drawdown"] > -0.30:
        score_fund += 30

    score_final = (
        score_fund * 0.7
        +
        score_quant * 0.3
    )

    return {
        "score_final": round(score_final, 2),
        "score_fund": score_fund,
        "score_quant": score_quant,
        "dy": round(ultimo["DY"], 2),
        "rsi": round(ultimo["RSI"], 2),
        "drawdown": round(
            ultimo["Drawdown"] * 100,
            2
        )
    }

# ==========================================
# RECOMENDAÇÃO
# ==========================================

def recomendacao(score):

    if score >= 70:
        return "MANTER"

    elif score >= 50:
        return "RESGATE PARCIAL"

    return "RESGATE TOTAL"

# ==========================================
# RESGATE
# ==========================================

def calcular_resgate(df_carteira, valor_resgate):

    carteira_ordenada = df_carteira.sort_values(
        by="Score",
        ascending=True
    )

    restante = valor_resgate

    sugestoes = []

    for _, row in carteira_ordenada.iterrows():

        if restante <= 0:
            break

        preco = row["Preço Atual"]

        qtd_disponivel = row["Disp. P/ Venda"]

        valor_total_ativo = (
            preco * qtd_disponivel
        )

        # quantidade sugerida
        qtd_venda = min(
            qtd_disponivel,
            int(restante // preco) + 1
        )

        subtotal = qtd_venda * preco

        if qtd_venda <= 0:
            continue

        comentario = ""

        if row["Score"] < 50:
            comentario = (
                "Baixo score e deterioração"
            )

        elif row["Score"] < 70:
            comentario = (
                "Tendência enfraquecida"
            )

        else:
            comentario = (
                "Resgate complementar"
            )

        sugestoes.append({
            "Ativo": row["Ativo"],
            "Quantidade": qtd_venda,
            "Preço": round(preco, 2),
            "Subtotal": round(subtotal, 2),
            "Comentário": comentario
        })

        restante -= subtotal

    return pd.DataFrame(sugestoes)

# ==========================================
# LEITURA CSV
# ==========================================

st.title("📈 Carteira B3")

try:

    carteira = pd.read_csv(ARQUIVO_CARTEIRA, sep=";")

    carteira["Ativo"] = (
        carteira["Ativo"]
        .astype(str)
        .str.strip()
        .str.replace("\u00A0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.upper()
    )
except Exception as e:

    st.error(f"Erro ao carregar CSV: {e}")

    st.stop()

# ==========================================
# PROCESSAMENTO
# ==========================================

resultado = []

with st.spinner("Processando ativos..."):

    for _, row in carteira.iterrows():

        ticker = row["Ativo"]

        try:

            historico, dividendos = baixar_historico(
                ticker
            )

            indicadores = calcular_indicadores(
                historico,
                dividendos
            )

            score = calcular_score(
                indicadores
            )

            resultado.append({
                "Ativo": ticker,
                "Descrição": row["Descrição"],
                "Carteira": row["Carteira"],
                "Preço Atual": row["Preço Atual"],
                "Qtd Total": row["Qtd Total"],
                "Valor Atual": row["Valor Atual"],
                "DY": score["dy"],
                "RSI": score["rsi"],
                "Drawdown %": score["drawdown"],
                "Score": score["score_final"],
                "Recomendação": recomendacao(
                    score["score_final"]
                )
            })

        except Exception as e:

            st.warning(
                f"Erro no ativo {ticker}: {e}"
            )

# ==========================================
# DATAFRAME FINAL
# ==========================================

df_resultado = pd.DataFrame(resultado)

# ==========================================
# DASHBOARD
# ==========================================

tab1, tab2 = st.tabs([
    "Carteira",
    "Resgate"
])

# ==========================================
# TAB CARTEIRA
# ==========================================

with tab1:

    st.subheader("📊 Análise da Carteira")

    st.dataframe(
        df_resultado.sort_values(
            by="Score",
            ascending=False
        ),
        use_container_width=True
    )

# ==========================================
# TAB RESGATE
# ==========================================

with tab2:

    st.subheader("💰 Planejamento de Resgate")

    valor_resgate = st.number_input(
        "Valor desejado para resgate",
        min_value=0.0,
        step=100.0
    )

    if st.button("Calcular Resgate"):

        sugestoes = calcular_resgate(
            df_resultado.merge(
                carteira,
                on="Ativo"
            ),
            valor_resgate
        )

        total = sugestoes["Subtotal"].sum()

        st.dataframe(
            sugestoes,
            use_container_width=True
        )

        st.success(
            f"Total sugerido: "
            f"R$ {total:,.2f}"
        )

# ==========================================
# RODAPÉ
# ==========================================

st.caption(
    "Sistema híbrido de análise "
    "fundamentalista + quantitativa"
)
