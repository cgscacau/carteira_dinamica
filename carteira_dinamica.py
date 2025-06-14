import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
import calendar
import io

# --- Configuração da Página ---
st.set_page_config(page_title="Otimizador de Carteira Inteligente", page_icon="💰", layout="wide")

# --- Funções de Busca e Otimização (sem alterações) ---
@st.cache_data(ttl=3600 * 4)
def buscar_indicadores_yfinance(tickers_sa):
    dados = []
    progress_bar = st.progress(0, text="Buscando indicadores fundamentalistas...")
    setor_map = {'Financial Services':'Financeiro','Utilities':'Elétrico/Saneamento','Basic Materials':'Materiais Básicos','Energy':'Energia','Consumer Cyclical':'Consumo Cíclico','Industrials':'Industrial','Real Estate':'Imobiliário','Communication Services':'Telecomunicações','Consumer Defensive':'Consumo Defensivo'}
    for i, ticker_str in enumerate(tickers_sa):
        try:
            ticker = yf.Ticker(ticker_str)
            info = ticker.info
            preco_atual = info.get('currentPrice', info.get('previousClose'))
            if not preco_atual or preco_atual == 0: continue
            divs_12m = ticker.dividends.last('365D').sum()
            dy = (divs_12m / preco_atual) if preco_atual > 0 else 0
            progress_bar.progress((i + 1) / len(tickers_sa), text=f"Buscando: {ticker_str}")
            if dy > 0:
                setor_br = setor_map.get(info.get('sector'), info.get('sector', 'N/A'))
                dados.append({'Ticker_yf': ticker_str, 'Ticker': ticker_str.replace('.SA', ''), 'Setor': setor_br, 'DY_Medio': dy, 'Preco_Atual': preco_atual})
        except Exception: pass
    progress_bar.empty()
    if not dados: return pd.DataFrame()
    return pd.DataFrame(dados).sort_values(by='DY_Medio', ascending=False).reset_index(drop=True)

def obter_lista_base_tickers():
    return ['PETR4','VALE3','BBAS3','GGBR4','BBDC4','ITSA4','CMIG4','CSNA3','BBSE3','CPLE6','TAEE11','ALUP11','SANB11','SAPR11','TRPL4','EGIE3','CSMG3','VIVT3','ABEV3','KLBN11','SUZB3','BEEF3','BRAP4','ELET3','SBSP3','AESB3','UNIP6','BRSR6','WIZC3','VBBR3','CPFE3','AURE3','ABCB4','CYRE3','DIRR3','LAVV3','MELK3','MTRE3','PLPL3','JHSF3','GOAU4','USIM5','CMIN3','PSSA3','TUPY3','KEPL3','ROMI3']

@st.cache_data
def obter_dados_historicos(tickers, period='5y'):
    dados, validos = [], []
    for t in tickers:
        try:
            ativo=yf.Ticker(t); hist=ativo.history(period=period,auto_adjust=True)
            if len(hist)<252: continue
            dados.append(hist['Close'].rename(t)); validos.append(t)
        except Exception: pass
    if not dados: return pd.DataFrame(), []
    return pd.concat(dados, axis=1).dropna(axis=0, how='all'), validos

@st.cache_data
def obter_dividendos_historicos(tickers, period='5y'):
    hist = {}
    for t in tickers:
        try:
            ativo=yf.Ticker(t); divs=ativo.dividends
            if not divs.empty:
                divs.index=divs.index.tz_localize(None)
                hist[t]=divs[divs.index >= pd.Timestamp.now()-pd.DateOffset(years=int(period[0]))]
        except Exception: pass
    return hist

def calcular_portfolio_otimizado(ret_esp, mat_cov):
    n=len(ret_esp)
    def neg_sharpe(w,r,c): p_r=np.sum(r*w); p_v=np.sqrt(np.dot(w.T,np.dot(c,w))); return -p_r/p_v if p_v!=0 else 0
    cons=({'type':'eq','fun':lambda x:np.sum(x)-1}); bnds=tuple((0,1) for _ in range(n))
    res=minimize(neg_sharpe,n*[1./n,],args=(ret_esp,mat_cov),method='SLSQP',bounds=bnds,constraints=cons)
    return res.x

@st.cache_data
def calcular_perfis_sazonais_individuais(tickers, dividendos_hist, period='5y'):
    perfis = {}
    num_anos = int(period[0])
    for ticker in tickers:
        perfil_mensal = np.zeros(12)
        if ticker in dividendos_hist and not dividendos_hist[ticker].empty:
            divs = dividendos_hist[ticker]
            media_mensal = divs.groupby(divs.index.month).sum() / num_anos
            for mes, valor in media_mensal.items():
                perfil_mensal[mes - 1] = valor
        perfis[ticker] = perfil_mensal
    return perfis

def otimizar_por_regularidade_dividendos(df_ativos, perfis_sazonais, valor_total_investir):
    tickers = df_ativos['Ticker_yf'].tolist()
    precos = df_ativos['Preco_Atual'].values
    n = len(tickers)
    def objetivo_regularidade(weights):
        qtd_acoes = (weights * valor_total_investir) / precos
        projecao_total = np.zeros(12)
        for i, ticker in enumerate(tickers):
            projecao_total += perfis_sazonais[ticker] * qtd_acoes[i]
        return np.std(projecao_total)
    cons=({'type':'eq','fun':lambda x:np.sum(x)-1}); bnds=tuple((0,1) for _ in range(n))
    res=minimize(objetivo_regularidade,n*[1./n,], method='SLSQP', bounds=bnds, constraints=cons)
    return res.x

def gerar_fronteira_eficiente(ret_esp, mat_cov, n=2000):
    num=len(ret_esp); res=np.zeros((3,n))
    for i in range(n):
        p=np.random.random(num); p/=np.sum(p)
        r,v=np.sum(ret_esp*p),np.sqrt(np.dot(p.T,np.dot(mat_cov,p)))
        res[0,i],res[1,i],res[2,i]=v,r,r/v
    return pd.DataFrame(res.T, columns=['Volatilidade','Retorno','SharpeRatio'])

def calcular_projecao_sazonal(df_carteira, perfis_individuais):
    proj={i:0 for i in range(1,13)}
    for ticker_yf, row in df_carteira.iterrows():
        qtd_acoes = row['Qtd_Acoes']
        perfil_ativo = perfis_individuais.get(ticker_yf, np.zeros(12))
        projecao_ativo = perfil_ativo * qtd_acoes
        for i in range(12):
            proj[i+1] += projecao_ativo[i]
            
    meses={i:n for i,n in enumerate(calendar.month_name) if i>0}
    df=pd.DataFrame(list(proj.items()),columns=['Mês_Num','Dividendo Projetado (R$)'])
    df['Mês']=df['Mês_Num'].map(meses); return df[['Mês','Dividendo Projetado (R$)']]

# --- Fluxo Principal da Aplicação ---
st.title("💰 Otimizador de Carteira Inteligente")
st.markdown("Defina seus objetivos: **retorno ajustado ao risco (Markowitz)**, **fluxo de caixa mensal (Regularidade)** ou **simples diversificação (Igualitária)**.")

df_base = buscar_indicadores_yfinance([t+'.SA' for t in obter_lista_base_tickers()])
if df_base.empty: st.error("Não foi possível carregar os indicadores. Tente recarregar a página."); st.stop()

setores_disponiveis = sorted(df_base['Setor'].unique())

with st.sidebar.form(key='parametros_form'):
    st.header("🔹 Filtros e Objetivos")
    valor_total_investir = st.number_input("Valor Total a Investir (R$)", min_value=1000.0, value=100000.0, step=1000.0, format="%.2f")
    num_ativos_considerar = st.slider("Universo de análise ('Top N' pagadoras de DY)", 10, len(df_base), 30, 5)
    sugestao_papeis = max(5, min(30, int(valor_total_investir / 7500)))
    num_papeis_carteira = st.slider("Número de ativos na carteira final", 2, num_ativos_considerar, sugestao_papeis, 1)
    setores_selecionados = st.multiselect("Setores Desejados", options=setores_disponiveis, default=setores_disponiveis)
    tipo_distribuicao = st.radio("Selecione o Objetivo da Otimização:", ('⚙️ Risco x Retorno (Markowitz)', '🗓️ Regularidade de Dividendos', '📊 Distribuição Igualitária'))
    analisar_button = st.form_submit_button(label="🚀 Analisar e Otimizar Carteira")

df_ativos_filtrados = df_base[df_base['Setor'].isin(setores_selecionados)].head(num_ativos_considerar)

if analisar_button:
    if df_ativos_filtrados.empty: st.error("Nenhum ativo encontrado."); st.stop()
    
    with st.spinner("Buscando dados históricos..."):
        precos_hist, tickers_validos = obter_dados_historicos(df_ativos_filtrados['Ticker_yf'].tolist())
    if precos_hist.empty or len(tickers_validos) < 2: st.error("Não foi possível obter dados históricos para 2 ou mais ativos."); st.stop()
        
    df_para_otimizar = df_ativos_filtrados[df_ativos_filtrados['Ticker_yf'].isin(tickers_validos)].set_index('Ticker_yf')
    
    dividendos_hist_completo = obter_dividendos_historicos(tickers_validos)
    perfis_sazonais = calcular_perfis_sazonais_individuais(tickers_validos, dividendos_hist_completo)

    if tipo_distribuicao == '⚙️ Risco x Retorno (Markowitz)':
        matriz_cov = precos_hist.pct_change().dropna().cov()*252; ret_esp = df_para_otimizar['DY_Medio'].reindex(matriz_cov.columns)
        with st.spinner("Otimizando por Markowitz..."): pesos_otimizados = calcular_portfolio_otimizado(ret_esp, matriz_cov)
        df_otimizado = pd.DataFrame({'Ticker_yf': ret_esp.index, 'Peso': pesos_otimizados})
    elif tipo_distribuicao == '🗓️ Regularidade de Dividendos':
        with st.spinner("Otimizando para regularidade..."):
            pesos_otimizados = otimizar_por_regularidade_dividendos(df_para_otimizar.reset_index(), perfis_sazonais, valor_total_investir)
        df_otimizado = pd.DataFrame({'Ticker_yf': df_para_otimizar.index, 'Peso': pesos_otimizados})
    else:
        df_otimizado = pd.DataFrame({'Ticker_yf': df_para_otimizar.index, 'Peso': 1/len(df_para_otimizar)})

    df_selecionado = df_otimizado.sort_values(by='Peso', ascending=False).head(num_papeis_carteira)
    df_selecionado['Peso'] = df_selecionado['Peso'] / df_selecionado['Peso'].sum()

    df_carteira = pd.merge(df_selecionado, df_base, on='Ticker_yf').set_index('Ticker_yf')
    precos_finais = precos_hist[df_carteira.index].iloc[-1]; df_carteira['Preco_Atual'] = precos_finais
    df_carteira['Valor_Alocado'] = df_carteira['Peso']*valor_total_investir; df_carteira['Qtd_Acoes'] = (df_carteira['Valor_Alocado']/df_carteira['Preco_Atual']).astype(int)
    
    pesos_finais=df_carteira['Peso'].values; retornos_finais=df_carteira['DY_Medio'].values
    matriz_cov_final = precos_hist[df_carteira.index].pct_change().dropna().cov() * 252
    ret_carteira_final, vol_carteira_final = np.sum(retornos_finais*pesos_finais), np.sqrt(np.dot(pesos_finais.T,np.dot(matriz_cov_final.values,pesos_finais)))

    df_projecao = calcular_projecao_sazonal(df_carteira, perfis_sazonais)
    dividendo_total_anual_projetado = df_projecao['Dividendo Projetado (R$)'].sum()

    st.header("✅ Carteira Otimizada"); col1,col2,col3 = st.columns(3)
    col1.metric("Yield Anual da Carteira", f"{ret_carteira_final*100:.2f}%")
    col2.metric("Dividendos Anuais Projetados", f"R$ {dividendo_total_anual_projetado:,.2f}")
    col3.metric("Volatilidade Anual da Carteira", f"{vol_carteira_final*100:.2f}%")
    st.dataframe(df_carteira[['Ticker', 'Setor', 'Peso', 'DY_Medio', 'Preco_Atual', 'Qtd_Acoes', 'Valor_Alocado']].sort_values(by='Peso',ascending=False).style.format({'Peso':'{:.2%}','DY_Medio':'{:.2%}','Preco_Atual':'R$ {:,.2f}','Valor_Alocado':'R$ {:,.2f}'}))
    
    # --- NOVO HEATMAP DE DIVIDENDOS ---
    st.markdown("---")
    st.header("🔥 Heatmap de Pagamento de Dividendos por Ativo")
    
    projecao_por_ativo = {}
    for ticker_yf, row in df_carteira.iterrows():
        perfil_ativo = perfis_sazonais.get(ticker_yf, np.zeros(12))
        projecao_por_ativo[row['Ticker']] = perfil_ativo * row['Qtd_Acoes']
        
    df_heatmap = pd.DataFrame.from_dict(projecao_por_ativo, orient='index')
    df_heatmap.columns = list(calendar.month_name)[1:]
    
    st.dataframe(df_heatmap.style.background_gradient(cmap='Greens', axis=1).format('R$ {:,.2f}'))
    st.caption("A tabela mostra o valor projetado de dividendos que cada ação da sua carteira pagará por mês. As cores mais intensas indicam os meses de maior probabilidade de pagamento para cada ativo.")
    
    st.markdown("---")
    if tipo_distribuicao == '⚙️ Risco x Retorno (Markowitz)':
        col_graf1, col_graf2 = st.columns([1.2, 1])
        with col_graf1:
            st.header("📈 Fronteira Eficiente"); retornos_diarios = precos_hist.pct_change().dropna(); matriz_cov_fronteira = retornos_diarios.cov()*252; retornos_esperados_fronteira = df_para_otimizar['DY_Medio'].reindex(matriz_cov_fronteira.columns); df_fronteira = gerar_fronteira_eficiente(retornos_esperados_fronteira, matriz_cov_fronteira); fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_fronteira.Volatilidade,y=df_fronteira.Retorno,mode='markers',marker=dict(size=5,color=df_fronteira.SharpeRatio,colorscale='Viridis',showscale=True,colorbar=dict(title='Índice Sharpe')),name='Carteiras Possíveis'))
            fig.add_trace(go.Scatter(x=[vol_carteira_final],y=[ret_carteira_final],mode='markers',marker=dict(color='red',size=15,symbol='star'),name='Sua Carteira')); fig.update_layout(xaxis_title='Volatilidade',yaxis_title='Retorno Esperado'); st.plotly_chart(fig, use_container_width=True)
        with col_graf2:
            st.header("📊 Projeção de Dividendos Mensais"); fig = go.Figure(go.Bar(x=df_projecao['Mês'],y=df_projecao['Dividendo Projetado (R$)'],text=df_projecao['Dividendo Projetado (R$)'].apply(lambda x:f'R${x:,.2f}'),textposition='outside')); fig.update_layout(xaxis_title='Mês',yaxis_title='Valor (R$)',xaxis={'categoryorder':'array','categoryarray':list(calendar.month_name)[1:]}); st.plotly_chart(fig, use_container_width=True)
    else:
        st.header("📊 Projeção de Dividendos Mensais"); fig = go.Figure(go.Bar(x=df_projecao['Mês'],y=df_projecao['Dividendo Projetado (R$)'],text=df_projecao['Dividendo Projetado (R$)'].apply(lambda x: f'R$ {x:,.2f}'),textposition='outside')); fig.update_layout(xaxis_title='Mês',yaxis_title='Valor (R$)',xaxis={'categoryorder':'array','categoryarray':list(calendar.month_name)[1:]}); st.plotly_chart(fig, use_container_width=True)
    
    st.header("🗓️ Tabela de Projeção Mensal"); st.dataframe(df_projecao.set_index('Mês').style.format({'Dividendo Projetado (R$)':'R$ {:,.2f}'}), use_container_width=True)
else:
    st.info("Ajuste os filtros na barra lateral e clique em 'Analisar e Otimizar' para começar.")
    st.subheader("Universo de Ativos Disponíveis (Dados do Yahoo Finance)"); st.dataframe(df_ativos_filtrados[['Ticker', 'Setor', 'DY_Medio', 'Preco_Atual']].style.format({'DY_Medio': '{:.2%}', 'Preco_Atual': 'R$ {:,.2f}'}))