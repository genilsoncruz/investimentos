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
# PARÂMETROS
# ==========================================

ARQUIVO_CSV = "carteira.csv"

# 0 = processa todos
QTD_ATIVOS_BUSCAR = 0

PERIODO_HISTORICO = "730d"

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

        c = re.sub(
            r"[^a-z0-9]+",
            "_",
            c
        )

        c = re.sub(
            r"_+",
            "_",
            c
        )

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

        if col not in df.columns:
            continue

        def tratar_numero(valor):

            if pd.isna(valor):
                return 0

            valor = str(valor).strip()

            if valor == "":
                return 0

            # =========================
            # FORMATO BR:
            # 1.234,56
            # =========================

            if "," in valor and "." in valor:

                valor = valor.replace(".", "")
                valor = valor.replace(",", ".")

            # =========================
            # FORMATO BR:
            # 123,45
            # =========================

            elif "," in valor:

                valor = valor.replace(",", ".")

            try:

                return float(valor)

            except:

                return 0

        df[col] = df[col].apply(
            tratar_numero
        )

    return df

# ==========================================
# CSV
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
def baixar_historico(
    ticker,
    periodo="730d"
):

    ticker_limpo = limpar_ticker(
        ticker
    )

    ticker_yf = (
        f"{ticker_limpo}.SA"
    )

    ativo = yf.Ticker(
        ticker_yf
    )

    historico = ativo.history(
        period=periodo,
        auto_adjust=True
    )

    if historico.empty:

        raise Exception(
            f"Sem dados para "
            f"{ticker_yf}"
        )

    dividendos = (
        ativo.dividends
    )

    return historico, dividendos

# ==========================================
# INDICADORES
# ==========================================

def calcular_indicadores(
    historico,
    dividendos
):

    mensal = (
        historico
        .resample("ME")
        .last()
    )

    qtd_meses = len(mensal)

    mensal["retorno"] = (
        mensal["Close"]
        .pct_change()
    )

    # =====================================
    # MÉDIAS MÓVEIS
    # =====================================

    mm_curta = min(
        9,
        max(2, qtd_meses // 2)
    )

    mm_longa = min(
        21,
        max(3, qtd_meses)
    )

    mensal["mm9"] = (
        SMAIndicator(
            mensal["Close"],
            window=mm_curta
        )
        .sma_indicator()
    )

    mensal["mm21"] = (
        SMAIndicator(
            mensal["Close"],
            window=mm_longa
        )
        .sma_indicator()
    )

    # =====================================
    # RSI
    # =====================================

    rsi_window = min(
        14,
        max(2, qtd_meses - 1)
    )

    mensal["rsi"] = (
        RSIIndicator(
            mensal["Close"],
            window=rsi_window
        )
        .rsi()
    )

    # =====================================
    # MOMENTUM
    # =====================================

    momentum_window = min(
        6,
        max(1, qtd_meses - 1)
    )

    mensal["momentum"] = (
        mensal["Close"]
        .pct_change(
            momentum_window
        )
    )

    # =====================================
    # DRAWDOWN
    # =====================================

    rolling_max = (
        mensal["Close"]
        .cummax()
    )

    mensal["drawdown"] = (
        mensal["Close"]
        - rolling_max
    ) / rolling_max

    # =====================================
    # VOLATILIDADE
    # =====================================

    vol_window = min(
        6,
        max(2, qtd_meses)
    )

    mensal["volatilidade"] = (

        mensal["retorno"]
        .rolling(vol_window)
        .std()

        * np.sqrt(12)
    )

    # =====================================
    # DIVIDENDOS
    # =====================================

    if dividendos.empty:

        mensal["dividendos"] = 0

    else:

        dividendos = (
            dividendos
            .resample("ME")
            .sum()
        )

        mensal["dividendos"] = (
            dividendos
        )

    mensal["dividendos"] = (
        mensal["dividendos"]
        .fillna(0)
    )

    div_window = min(
        12,
        max(1, qtd_meses)
    )

    mensal["dividendos_12m"] = (

        mensal["dividendos"]
        .rolling(div_window)
        .sum()
    )

    mensal["dy"] = (

        mensal["dividendos_12m"]

        / mensal["Close"]

    ) * 100

    # =====================================
    # QUALIDADE HISTÓRICO
    # =====================================

    qualidade = "ALTA"

    if qtd_meses < 12:

        qualidade = "BAIXA"

    elif qtd_meses < 24:

        qualidade = "MEDIA"

    mensal[
        "qualidade_historico"
    ] = qualidade

    return mensal

# ==========================================
# SCORE
# ==========================================

def calcular_score(df):

    ultimo = df.iloc[-1]

    score_quant = 0

    if (
        pd.notnull(
            ultimo["mm9"]
        )
        and
        pd.notnull(
            ultimo["mm21"]
        )
    ):

        if (
            ultimo["mm9"]
            >
            ultimo["mm21"]
        ):

            score_quant += 30

    if pd.notnull(
        ultimo["rsi"]
    ):

        if (
            40
            <=
            ultimo["rsi"]
            <=
            70
        ):

            score_quant += 20

    if pd.notnull(
        ultimo["momentum"]
    ):

        if (
            ultimo["momentum"]
            > 0
        ):

            score_quant += 20

    if pd.notnull(
        ultimo["drawdown"]
    ):

        if (
            ultimo["drawdown"]
            > -0.20
        ):

            score_quant += 20

    if pd.notnull(
        ultimo["volatilidade"]
    ):

        if (
            ultimo["volatilidade"]
            < 0.30
        ):

            score_quant += 10

    # =====================================
    # SCORE FUNDAMENTALISTA
    # =====================================

    score_fund = 0

    dy = ultimo["dy"]

    if pd.notnull(dy):

        if dy >= 10:

            score_fund += 40

        elif dy >= 6:

            score_fund += 30

        elif dy >= 4:

            score_fund += 20

    if pd.notnull(
        ultimo["dividendos_12m"]
    ):

        if (
            ultimo["dividendos_12m"]
            > 0
        ):

            score_fund += 30

    if pd.notnull(
        ultimo["drawdown"]
    ):

        if (
            ultimo["drawdown"]
            > -0.30
        ):

            score_fund += 30

    score_final = (

        score_fund * 0.7

        +

        score_quant * 0.3
    )

    # =====================================
    # PENALIZA HISTÓRICO
    # =====================================

    qualidade = ultimo.get(
        "qualidade_historico",
        "ALTA"
    )

    if qualidade == "BAIXA":

        score_final *= 0.7

    elif qualidade == "MEDIA":

        score_final *= 0.85

    return {

        "dy": round(
            float(
                ultimo["dy"]
            ),
            2
        ) if pd.notnull(
            ultimo["dy"]
        ) else 0,

        "rsi": round(
            float(
                ultimo["rsi"]
            ),
            2
        ) if pd.notnull(
            ultimo["rsi"]
        ) else 0,

        "drawdown": round(
            float(
                ultimo["drawdown"]
                * 100
            ),
            2
        ) if pd.notnull(
            ultimo["drawdown"]
        ) else 0,

        "score_quant":
            round(
                score_quant,
                2
            ),

        "score_fund":
            round(
                score_fund,
                2
            ),

        "score_final":
            round(
                score_final,
                2
            ),

        "qualidade_historico":
            qualidade
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

    df = df_carteira.copy()

    # =====================================
    # PRIORIDADE
    # =====================================

    prioridade = {

        "RESGATE TOTAL": 1,

        "RESGATE PARCIAL": 2,

        "MANTER": 3
    }

    df["prioridade"] = (
        df["recomendacao"]
        .map(prioridade)
    )

    # =====================================
    # ORDENAÇÃO
    # =====================================

    df = df.sort_values(
        by=[
            "prioridade",
            "score"
        ],
        ascending=[True, True]
    )

    restante = valor_resgate

    sugestoes = []

    # =====================================
    # LOOP
    # =====================================

    for _, row in df.iterrows():

        if restante <= 0:
            break

        # =====================================
        # PREÇO
        # =====================================

        preco = row.get(
            "preco_atual",
            0
        )

        if pd.isna(preco):
            continue

        preco = float(preco)

        if preco <= 0:
            continue

        # =====================================
        # QUANTIDADE DISPONÍVEL
        # =====================================

        qtd_disponivel = row.get(
            "disp_p_venda",
            0
        )

        if pd.isna(
            qtd_disponivel
        ):

            qtd_disponivel = 0

        qtd_disponivel = int(
            qtd_disponivel
        )

        if qtd_disponivel <= 0:
            continue

        # =====================================
        # QUANTIDADE NECESSÁRIA
        # =====================================

        qtd_necessaria = int(
            restante // preco
        )

        if restante % preco > 0:
            qtd_necessaria += 1

        qtd_venda = min(
            qtd_disponivel,
            qtd_necessaria
        )

        if qtd_venda <= 0:
            continue

        subtotal = (
            qtd_venda * preco
        )

        # =====================================
        # COMENTÁRIO
        # =====================================

        recomendacao = row.get(
            "recomendacao",
            "MANTER"
        )

        if recomendacao == "RESGATE TOTAL":

            comentario = (
                "Ativo com "
                "deterioração estrutural."
            )

        elif recomendacao == "RESGATE PARCIAL":

            comentario = (
                "Ativo com "
                "enfraquecimento "
                "de tendência."
            )

        else:

            comentario = (
                "Utilizado para "
                "complementar o "
                "valor do resgate."
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
                recomendacao,

            "Comentário":
                comentario
        })

        restante -= subtotal

    # =====================================
    # RESULTADO
    # =====================================

    return pd.DataFrame(
        sugestoes
    )

# ==========================================
# TÍTULO
# ==========================================

st.title(
    "📈 Carteira B3"
)

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
# LIMITA PROCESSAMENTO
# ==========================================

if QTD_ATIVOS_BUSCAR > 0:

    carteira = (
        carteira
        .head(
            QTD_ATIVOS_BUSCAR
        )
    )

# ==========================================
# PROCESSAMENTO
# ==========================================

resultado = []

ativos_historico_ruim = []

progress_bar = st.progress(0)

status_text = st.empty()

total_ativos = len(carteira)

for i, (_, row) in enumerate(
    carteira.iterrows()
):

    ticker = row["ativo"]

    progresso = (
        (i + 1)
        / total_ativos
    )

    progress_bar.progress(
        progresso
    )

    status_text.write(
        f"""
        Processando:
        {ticker}
        ({i+1}/{total_ativos})
        """
    )

    try:

        historico, dividendos = (
            baixar_historico(
                ticker,
                PERIODO_HISTORICO
            )
        )

        indicadores = (
            calcular_indicadores(
                historico,
                dividendos
            )
        )

        score = (
            calcular_score(
                indicadores
            )
        )

        if (
            score[
                "qualidade_historico"
            ]
            !=
            "ALTA"
        ):

            ativos_historico_ruim.append({

                "Ativo":
                    ticker,

                "Qualidade":
                    score[
                        "qualidade_historico"
                    ]
            })

        resultado.append({

            "ativo":
                ticker,

            "descricao":
                row.get(
                    "descricao",
                    ""
                ),

            "carteira":
                row.get(
                    "carteira",
                    ""
                ),

            "preco_atual":
                row.get(
                    "preco_atual",
                    0
                ),

            "valor_atual":
                row.get(
                    "valor_atual",
                    0
                ),

            "qtd_total":
                row.get(
                    "qtd_total",
                    0
                ),

            "disp_p_venda":
                row.get(
                    "disp_p_venda",
                    0
                ),

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

            "qualidade_historico":
                score[
                    "qualidade_historico"
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

progress_bar.empty()

status_text.empty()

# ==========================================
# DATAFRAME FINAL
# ==========================================

df_resultado = pd.DataFrame(
    resultado
)

if df_resultado.empty:

    st.error(
        "Nenhum ativo "
        "foi processado."
    )

    st.stop()

# ==========================================
# ORDENAÇÃO
# ==========================================

df_ordenado = (

    df_resultado

    .sort_values(
        by="score",
        ascending=False
    )
)

# ==========================================
# TABS
# ==========================================

tab1, tab2, tab3 = st.tabs([

    "📊 Carteira",

    "💰 Resgate",

    "⚠ Histórico"
])

# ==========================================
# TAB CARTEIRA
# ==========================================

with tab1:

    st.subheader(
        "Análise da Carteira"
    )

    st.dataframe(
        df_ordenado[
            [

                "ativo",

                "descricao",

                "qtd_total",

                "preco_atual",

                "valor_atual",

                "dy",

                "rsi",

                "drawdown",

                "score",

                "qualidade_historico",

                "recomendacao"
            ]
        ],
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
                df_resultado.copy(),
                valor_resgate
            )
        )

        if sugestoes.empty:

            st.warning(
                """
                Nenhum ativo
                com indicação de
                resgate encontrado.
                """
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
# TAB HISTÓRICO
# ==========================================

with tab3:

    st.subheader(
        "Ativos com histórico insuficiente"
    )

    if (
        len(
            ativos_historico_ruim
        )
        == 0
    ):

        st.success(
            """
            Nenhum ativo
            com problema
            de histórico.
            """
        )

    else:

        st.dataframe(
            pd.DataFrame(
                ativos_historico_ruim
            ),
            use_container_width=True
        )

# ==========================================
# RODAPÉ
# ==========================================

st.caption(
    "Sistema híbrido "
    "fundamentalista + "
    "quantitativo"
)
