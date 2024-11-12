import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.grid import grid


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

def build_main(tickers, prices):
    weights = np.ones(len(tickers)) / len(tickers)
    portfolio_values = prices[tickers] @ weights
    prices_with_portfolio = prices.assign(portfolio=portfolio_values)

    norm_prices = 100 * prices_with_portfolio / prices_with_portfolio.iloc[0]
    returns = prices_with_portfolio.pct_change()[1:]
    vols = returns.std() * np.sqrt(252)
    rets = (norm_prices.iloc[-1] - 100) / 100

    # Obter dados adicionais de cada ativo
    additional_data = get_additional_data(tickers)

    # Ajuste de layout para exibir ícone e nome do ticker na mesma linha
    mygrid = grid(5, 5, 5, 5, 5, 5, vertical_align="top")
    for t in prices_with_portfolio.columns:
        c = mygrid.container(border=True)
        col_image, col_text = c.columns([1, 4])

        # Exibir ícone ao lado do nome do ticker
        if t == "portfolio":
            col_image.image("images/pie-chart-dollar-svgrepo-com.svg", width=60)
            col_text.subheader("Portfolio", divider="red")
        elif t == "IBOV":
            col_image.image("images/pie-chart-svgrepo-com.svg", width=60)
            col_text.subheader("IBOV", divider="red")
        else:
            col_image.image(f'https://raw.githubusercontent.com/thefintz/icones-b3/main/icones/{t}.png', width=60)
            col_text.subheader(t, divider="red")

        # Exibir cada métrica individualmente, sem utilizar DataFrame
        with c:
            st.write("Retorno:", f"{rets.get(t, 0):.0%}")
            st.write("Volatilidade:", f"{vols.get(t, 0):.0%}")
            st.write("Dividend Yield:", f"{additional_data.loc[t, 'Dividend Yield']:.2f}%" if t in additional_data.index else "N/A")
            st.write("Preço Atual:", f"R${additional_data.loc[t, 'Preço Atual']:.2f}" if t in additional_data.index else "N/A")

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


st.set_page_config(layout="wide")

with st.sidebar:
    tickers, prices = build_sidebar()

st.title('Análise de parâmetros de ações')

if tickers:
    build_main(tickers, prices)
