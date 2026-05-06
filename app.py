# ==========================================
# app.py
# Sistema de análise de carteira B3
# Streamlit Cloud Ready
# ==========================================

import re
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

# ==========================================
# CONFIG STREAMLIT
# ==========================================

st.set_page_config(
    page_title="Carteira B3",
    layout="wide"
)

# ==========================================
# CONFIG
# ==========================================

ARQUIVO_CSV = "carteira.csv"

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================

def limpar_ticker(ticker):

    ticker = (
        str(ticker)
        .strip()
        .replace("\u00A0", "")
        .replace(" ", "")
        .upper()
    )

    ticker = ticker.replace(".SA", "")

    return ticker


def normalizar_colunas(df):

    novas_colunas = []

    for c in df.columns:

        c = (
            str(c)
            .lower()
            .strip()
            .replace("ç", "c")
            .replace("ã", "a")
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("%", "pct")
        )

        # substitui caracteres especiais
        c = re.sub(r"[^a-z0-9]+", "_", c)

        # remove múltiplos _
        c = re.sub(r"_+", "_", c)

        c = c.strip("_")

        novas_colunas.append(c)

    df.columns = novas_colunas

    return df


def converter_numeros(df):

    colunas_numericas = [
        "preco_atual",
        "valor_atual",
        "variacao_pct",
        "qtd_total",
        "qtd_bloq",
        "qtd_pend",
        "qtd_exec",
        "disp_p_venda"
    ]

    for col in colunas_numericas:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df

# ==========================================
# LEITURA CSV
# ==========================================

@st.cache_data
def carregar_carteira():

    df = pd.read_csv(
        ARQUIVO_CSV,
        sep=";",
        encoding="utf-8-sig"
    )

    df = normalizar_colunas(df)

    df = converter_numeros(df)

    if "ativo" not in df.columns:

        raise Exception(
            "Coluna 'ativo' não encontrada."
        )

    # limpa ticker
    df["ativo"] = (
        df["ativo"]
        .astype(str)
        .apply(limpar_ticker)
    )

    return df

# ==========================================
# YAHOO FINANCE
# ==========================================

@st.cache_data(ttl=3600)
def baixar_historico(ticker, periodo="2y"):

    ticker_limpo = limpar_ticker(ticker)

    ticker_yf = f"{ticker_limpo}.SA"

    ativo = yf.Ticker(ticker_yf)

    historico = ativo.history(
        period=periodo,
        auto_adjust=True
    )

    if historico.empty:

        raise Exception(
            f"Sem dados para {ticker_yf}"
        )

    dividendos = ativo.dividends

    return historico, dividendos

# ==========================================
# INDICADORES
# ==========================================

def calcular_indicadores(historico, dividendos):

    mensal = (
        historico
        .resample("M")
        .last()
    )

    mensal["retorno"] = (
        mensal["Close"].pct_change()
    )

    # MM
    mensal["mm9"] = SMAIndicator(
        mensal["Close"],
        window=9
    ).sma_indicator()

    mensal["mm21"] = SMAIndicator(
        mensal["Close"],
        window=21
    ).sma_indicator()

    # RSI
    mensal["rsi"] = RSIIndicator(
        mensal["Close"],
        window=14
    ).rsi()

    # Momentum
    mensal["momentum"] = (
        mensal["Close"]
        .pct_change(6)
    )

    # Drawdown
    rolling_max = (
        mensal["Close"]
        .cummax()
    )

    mensal["drawdown"] = (
        mensal["Close"]
        - rolling_max
    ) / rolling_max

    # Volatilidade
    mensal["volatilidade"] = (
        mensal["retorno"]
        .rolling(6)
        .std()
        * np.sqrt(12)
    )

    # =====================
    # DIVIDENDOS
    # =====================

    if dividendos.empty:

        mensal["dividendos"] = 0

    else:

        dividendos = (
            dividendos
            .resample("M")
            .sum()
        )

        mensal["dividendos"] = dividendos

    mensal["dividendos"] = (
        mensal["dividendos"]
        .fillna(0)
    )

    mensal["dividendos_12m"] = (
        mensal["dividendos"]
        .rolling(12)
        .sum()
    )

    mensal["dy"] = (
        mensal["dividendos_12m"]
        / mensal["Close"]
    ) * 100

    return mensal

# ==========================================
# SCORE
# ==========================================

def calcular_score(df):

    ultimo = df.iloc[-1]

    # =====================
    # SCORE QUANT
    # =====================

    score_quant = 0

    # tendência
    if (
        pd.notnull(ultimo["mm9"])
        and
        pd.notnull(ultimo["mm21"])
    ):

        if ultimo["mm9"] > ultimo["mm21"]:
            score_quant += 30

    # RSI
    if pd.notnull(ultimo["rsi"]):

        if 40 <= ultimo["rsi"] <= 70:
            score_quant += 20

    # momentum
    if pd.notnull(ultimo["momentum"]):

        if ultimo["momentum"] > 0:
            score_quant += 20

    # drawdown
    if pd.notnull(ultimo["drawdown"]):

        if ultimo["drawdown"] > -0.20:
            score_quant += 20

    # volatilidade
    if pd.notnull(ultimo["volatilidade"]):

        if ultimo["volatilidade"] < 0.30:
            score_quant += 10

    # =====================
    # SCORE FUND
    # =====================

    score_fund = 0

    dy = ultimo["dy"]

    if pd.notnull(dy):

        if dy >= 10:
            score_fund += 40

        elif dy >= 6:
            score_fund += 30

        elif dy >= 4:
            score_fund += 20

    # dividendos consistentes
    if pd.notnull(
        ultimo["dividendos_12m"]
    ):

        if ultimo["dividendos_12m"] > 0:
            score_fund += 30

    # drawdown moderado
    if pd.notnull(
        ultimo["drawdown"]
    ):

        if ultimo["drawdown"] > -0.30:
            score_fund += 30

    # =====================
    # SCORE FINAL
    # =====================

    score_final = (
        score_fund * 0.7
        +
        score_quant * 0.3
    )

    return {

        "dy": round(
            float(ultimo["dy"]),
            2
        ) if pd.notnull(ultimo["dy"]) else 0,

        "rsi": round(
            float(ultimo["rsi"]),
            2
        ) if pd.notnull(ultimo["rsi"]) else 0,

        "drawdown": round(
            float(
                ultimo["drawdown"] * 100
            ),
            2
        ) if pd.notnull(
            ultimo["drawdown"]
        ) else 0,

        "score_quant":
            round(score_quant, 2),

        "score_fund":
            round(score_fund, 2),

        "score_final":
            round(score_final, 2)
    }

# ==========================================
# RECOMENDAÇÃO
# ==========================================

def gerar_recomendacao(score):

    if score >= 70:
        return "MANTER"

    elif score >= 50:
        return "RESGATE PARCIAL"

    return "RESGATE TOTAL"

# ==========================================
# RESGATE
# ==========================================

def calcular_resgate(
    df_carteira,
    valor_resgate
):

    carteira_ordenada = (
        df_carteira
        .sort_values(
            by="score",
            ascending=True
        )
    )

    restante = valor_resgate

    sugestoes = []

    for _, row in carteira_ordenada.iterrows():

        if restante <= 0:
            break

        preco = float(
            row["preco_atual"]
        )

        qtd_disponivel = int(
            row["disp_p_venda"]
        )

        if qtd_disponivel <= 0:
            continue

        qtd_venda = min(
            qtd_disponivel,
            int(restante // preco) + 1
        )

        subtotal = (
            qtd_venda * preco
        )

        if subtotal <= 0:
            continue

        # comentário
        if row["score"] < 50:

            comentario = (
                "Baixo score e "
                "deterioração"
            )

        elif row["score"] < 70:

            comentario = (
                "Tendência "
                "enfraquecida"
            )

        else:

            comentario = (
                "Resgate complementar"
            )

        sugestoes.append({

            "Ativo":
                row["ativo"],

            "Quantidade":
                qtd_venda,

            "Preço":
                round(preco, 2),

            "Subtotal":
                round(subtotal, 2),

            "Score":
                round(
                    row["score"],
                    2
                ),

            "Recomendação":
                row[
                    "recomendacao"
                ],

            "Comentário":
                comentario
        })

        restante -= subtotal

    return pd.DataFrame(sugestoes)

# ==========================================
# TÍTULO
# ==========================================

st.title("📈 Carteira B3")

# ==========================================
# CARREGA CARTEIRA
# ==========================================

try:

    carteira = carregar_carteira()

except Exception as e:

    st.error(
        f"Erro ao carregar CSV: {e}"
    )

    st.stop()

# ==========================================
# DEBUG
# ==========================================

with st.expander("DEBUG"):

    st.write("Colunas:")
    st.write(
        carteira.columns.tolist()
    )

    st.write("Amostra:")
    st.dataframe(
        carteira.head()
    )

# ==========================================
# PROCESSAMENTO
# ==========================================

resultado = []

with st.spinner(
    "Processando ativos..."
):

    for _, row in carteira.iterrows():

        ticker = row["ativo"]

        try:

            st.write(
                f"Processando: {ticker}"
            )

            historico, dividendos = (
                baixar_historico(
                    ticker
                )
            )

            indicadores = (
                calcular_indicadores(
                    historico,
                    dividendos
                )
            )

            score = calcular_score(
                indicadores
            )

            resultado.append({

                "ativo":
                    ticker,

                "descricao":
                    row[
                        "descricao"
                    ],

                "carteira":
                    row[
                        "carteira"
                    ],

                "preco_atual":
                    row[
                        "preco_atual"
                    ],

                "valor_atual":
                    row[
                        "valor_atual"
                    ],

                "qtd_total":
                    row[
                        "qtd_total"
                    ],

                "disp_p_venda":
                    row[
                        "disp_p_venda"
                    ],

                "dy":
                    score["dy"],

                "rsi":
                    score["rsi"],

                "drawdown":
                    score[
                        "drawdown"
                    ],

                "score_quant":
                    score[
                        "score_quant"
                    ],

                "score_fund":
                    score[
                        "score_fund"
                    ],

                "score":
                    score[
                        "score_final"
                    ],

                "recomendacao":
                    gerar_recomendacao(
                        score[
                            "score_final"
                        ]
                    )
            })

        except Exception as e:

            st.error(
                f"""
                ERRO NO ATIVO:
                {ticker}

                {str(e)}
                """
            )

            st.code(
                traceback.format_exc()
            )

            continue

# ==========================================
# DATAFRAME FINAL
# ==========================================

df_resultado = pd.DataFrame(
    resultado
)

# ==========================================
# PROTEÇÃO DF VAZIO
# ==========================================

st.write(
    "Quantidade processada:",
    len(df_resultado)
)

if df_resultado.empty:

    st.error(
        "Nenhum ativo foi "
        "processado."
    )

    st.stop()

# ==========================================
# ORDENAÇÃO
# ==========================================

if "score" in df_resultado.columns:

    df_ordenado = (
        df_resultado
        .sort_values(
            by="score",
            ascending=False
        )
    )

else:

    df_ordenado = (
        df_resultado.copy()
    )

# ==========================================
# TABS
# ==========================================

tab1, tab2 = st.tabs([
    "📊 Carteira",
    "💰 Resgate"
])

# ==========================================
# TAB CARTEIRA
# ==========================================

with tab1:

    st.subheader(
        "Análise da Carteira"
    )

    st.dataframe(
        df_ordenado,
        use_container_width=True
    )

# ==========================================
# TAB RESGATE
# ==========================================

with tab2:

    st.subheader(
        "Planejamento de Resgate"
    )

    valor_resgate = (
        st.number_input(
            "Valor desejado "
            "para resgate",
            min_value=0.0,
            step=100.0
        )
    )

    if st.button(
        "Calcular Resgate"
    ):

        sugestoes = (
            calcular_resgate(
                df_resultado,
                valor_resgate
            )
        )

        if sugestoes.empty:

            st.warning(
                "Nenhuma sugestão "
                "encontrada."
            )

        else:

            total = (
                sugestoes[
                    "Subtotal"
                ].sum()
            )

            st.dataframe(
                sugestoes,
                use_container_width=True
            )

            st.success(
                f"""
                Total sugerido:
                R$ {total:,.2f}
                """
            )

# ==========================================
# RODAPÉ
# ==========================================

st.caption(
    "Sistema híbrido "
    "fundamentalista + "
    "quantitativo"
)
