import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import sqlite3
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid

# Função para configurar o banco de dados SQLite e carregar dados persistentes
def initialize_database():
    conn = sqlite3.connect('tickers_data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS tickers_data (
                        ticker TEXT PRIMARY KEY,
                        retorno TEXT,
                        volatilidade TEXT,
                        dividend_yield TEXT,
                        preco_atual TEXT,
                        icon_url TEXT)''')
    conn.commit()
    return conn, cursor

# Função para salvar dados no banco
def save_data(cursor, conn, ticker_data):
    for ticker, data in ticker_data.items():
        cursor.execute('''INSERT OR REPLACE INTO tickers_data (ticker, retorno, volatilidade, dividend_yield, preco_atual, icon_url)
                          VALUES (?, ?, ?, ?, ?, ?)''', 
                       (ticker, data['Retorno'], data['Volatilidade'], data['Dividend Yield'], data['Preço Atual'], data['Icon URL']))
    conn.commit()

# Função para carregar dados do banco
def load_data(cursor):
    cursor.execute('SELECT * FROM tickers_data')
    data = cursor.fetchall()
    return {
        row[0]: {
            'Retorno': row[1],
            'Volatilidade': row[2],
            'Dividend Yield': row[3],
            'Preço Atual': row[4],
            'Icon URL': row[5]
        }
        for row in data
    }

def build_sidebar():
    st.image("images/logo_2.png")
    ticker_list = pd.read_csv("tickers_ibra.csv", index_col=0)
    tickers = st.multiselect(label="Selecione as Empresas", options=ticker_list, placeholder='Códigos')
    
    if not tickers:
        return None, None
    
    tickers_with_suffix = [t + ".SA" for t in tickers]
    start_date = st.date_input("De", value=datetime(2023, 1, 2))
    end_date = st.date_input("Até", value=datetime.today())

    # Download de dados históricos
    prices = yf.download(tickers_with_suffix, start=start_date, end=end_date)["Adj Close"]

    # Se houver apenas um ticker, prices será uma Série. Convertê-la para DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
        prices.columns = [tickers[0]]

    # Limpeza de colunas para remover o sufixo ".SA"
    prices.columns = prices.columns.str.rstrip(".SA")
    
    # Baixar dados do IBOV
    prices['IBOV'] = yf.download("^BVSP", start=start_date, end=end_date)["Adj Close"]

    return tickers, prices

def get_additional_data(tickers):
    additional_data = {}
    for ticker in tickers:
        stock_info = yf.Ticker(f"{ticker}.SA").info
        additional_data[ticker] = {
            'Dividend Yield': stock_info.get('dividendYield', 0) * 100,
            'Preço Atual': stock_info.get('currentPrice', None),
            'Rentabilidade': stock_info.get('returnOnEquity', 0) * 100
        }
    return pd.DataFrame(additional_data).T

def build_main(tickers, prices, stored_data):
    weights = np.ones(len(tickers)) / len(tickers)
    portfolio_values = prices[tickers] @ weights
    prices_with_portfolio = prices.assign(portfolio=portfolio_values)

    norm_prices = 100 * prices_with_portfolio / prices_with_portfolio.iloc[0]
    returns = prices_with_portfolio.pct_change()[1:]
    vols = returns.std() * np.sqrt(252)
    rets = (norm_prices.iloc[-1] - 100) / 100

    additional_data = get_additional_data(tickers)

    ticker_data = {}
    for t in prices_with_portfolio.columns:
        icon_url = (
            "images/pie-chart-dollar-svgrepo-com.svg" if t == "portfolio" 
            else "images/pie-chart-svgrepo-com.svg" if t == "IBOV" 
            else f'https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{t}.png'
        )
        ticker_data[t] = {
            "Retorno": f"{rets.get(t, 0):.0%}",
            "Volatilidade": f"{vols.get(t, 0):.0%}",
            "Dividend Yield": f"{additional_data.loc[t, 'Dividend Yield']:.2f}%" if t in additional_data.index else "N/A",
            "Preço Atual": f"R${additional_data.loc[t, 'Preço Atual']:.2f}" if t in additional_data.index else "N/A",
            "Icon URL": icon_url
        }

    save_data(cursor, conn, ticker_data)
    stored_data.update(load_data(cursor))

    mygrid = grid(5, 5, 5, 5, 5, 5, vertical_align="top")
    for t, data in stored_data.items():
        c = mygrid.container(border=True)
        col_image, col_text = c.columns([1, 4])

        # Exibir ícone ao lado do nome do ticker
        col_image.image(data['Icon URL'], width=60)
        col_text.subheader(t, divider="red")

        # Função para exibir métricas com divisores
        def display_metric(label, value):
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; padding: 5px 0;">
                    <span>{label}:</span> <span>{value}</span>
                </div>
                <hr style="border: none; border-top: 1px solid #ccc; margin: 5px 0;">
                """,
                unsafe_allow_html=True
            )

        # Exibir cada métrica individualmente com divisores
        with c:
            display_metric("Retorno", data['Retorno'])
            display_metric("Volatilidade", data['Volatilidade'])
            display_metric("Dividend Yield", data['Dividend Yield'])
            display_metric("Preço Atual", data['Preço Atual'])

        style_metric_cards(background_color='rgba(255,255,255,0)')

    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.subheader("Desempenho Relativo")
        st.line_chart(norm_prices, height=600)

    with col2:
        st.subheader("Risco-Retorno")
        fig = px.scatter(
            x=vols,
            y=rets,
            text=vols.index,
            color=rets/vols,
            color_continuous_scale=px.colors.sequential.Bluered_r
        )
        fig.update_traces(
            textfont_color='white',
            marker=dict(size=45),
            textfont_size=10,
        )
        fig.layout.yaxis.title = 'Retorno Total'
        fig.layout.xaxis.title = 'Volatilidade (anualizada)'
        fig.layout.height = 600
        fig.layout.xaxis.tickformat = ".0%"
        fig.layout.yaxis.tickformat = ".0%"        
        fig.layout.coloraxis.colorbar.title = 'Sharpe'
        st.plotly_chart(fig, use_container_width=True)

# Configuração inicial do Streamlit e do banco de dados
st.set_page_config(layout="wide")
conn, cursor = initialize_database()
stored_data = load_data(cursor)

with st.sidebar:
    tickers, prices = build_sidebar()

st.title('Análise de parâmetros de ações')

# Carregar dados salvos automaticamente se nenhum ticker for selecionado no momento do carregamento da página
if not tickers and stored_data:
    tickers = list(stored_data.keys())
    prices = None  # Evitar download de preços ao carregar dados do banco

if tickers:
    build_main(tickers, prices, stored_data)
