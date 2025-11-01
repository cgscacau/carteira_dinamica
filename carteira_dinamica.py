import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ativos_b3

# ==================== CONFIGURAÇÃO ====================
st.set_page_config(
    page_title="Otimizador de Carteira - Markowitz",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== FUNÇÕES ====================

def baixar_dados_ativos(tickers, data_inicio, data_fim):
    """Baixa dados de múltiplos ativos"""
    dados_dict = {}
    sucessos = []
    erros = []
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    for i, ticker in enumerate(tickers):
        status.text(f"📥 Baixando {ticker}... ({i+1}/{len(tickers)})")
        try:
            ativo = yf.Ticker(ticker)
            hist = ativo.history(start=data_inicio, end=data_fim)
            
            if not hist.empty and len(hist) >= 20:
                dados_dict[ticker] = hist['Close']
                sucessos.append(ticker)
            else:
                erros.append(ticker)
        except:
            erros.append(ticker)
        
        progress_bar.progress((i + 1) / len(tickers))
    
    progress_bar.empty()
    status.empty()
    
    return dados_dict, sucessos, erros

def baixar_dividendos(tickers, data_inicio, data_fim):
    """Baixa dados de dividendos"""
    dividendos_dict = {}
    
    for ticker in tickers:
        try:
            ativo = yf.Ticker(ticker)
            divs = ativo.dividends
            
            if not divs.empty:
                divs = divs[(divs.index >= pd.Timestamp(data_inicio)) & 
                           (divs.index <= pd.Timestamp(data_fim))]
                if not divs.empty:
                    dividendos_dict[ticker] = divs
        except:
            pass
    
    return dividendos_dict

def calcular_metricas_portfolio(pesos, retornos, matriz_cov, taxa_livre_risco):
    """Calcula métricas de um portfolio"""
    ret = np.sum(retornos * pesos) * 252
    vol = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos))) * np.sqrt(252)
    sharpe = (ret - taxa_livre_risco) / vol if vol > 0 else 0
    
    return ret, vol, sharpe

def otimizar_sharpe(retornos, matriz_cov, taxa_livre_risco):
    """Otimiza para máximo Sharpe Ratio"""
    num_ativos = len(retornos)
    
    def objetivo(pesos):
        ret, vol, sharpe = calcular_metricas_portfolio(pesos, retornos, matriz_cov, taxa_livre_risco)
        return -sharpe
    
    restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limites = tuple((0, 1) for _ in range(num_ativos))
    inicial = np.array([1/num_ativos] * num_ativos)
    
    resultado = minimize(objetivo, inicial, method='SLSQP', 
                        bounds=limites, constraints=restricoes)
    
    return resultado.x

def otimizar_minima_volatilidade(matriz_cov):
    """Otimiza para mínima volatilidade"""
    num_ativos = len(matriz_cov)
    
    def objetivo(pesos):
        return np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos))) * np.sqrt(252)
    
    restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limites = tuple((0, 1) for _ in range(num_ativos))
    inicial = np.array([1/num_ativos] * num_ativos)
    
    resultado = minimize(objetivo, inicial, method='SLSQP',
                        bounds=limites, constraints=restricoes)
    
    return resultado.x

# ==================== INICIALIZAÇÃO ====================

if 'ativos_finais' not in st.session_state:
    st.session_state.ativos_finais = []

# ==================== HEADER ====================

st.markdown('<p class="main-header">📊 Otimizador de Carteira - Teoria de Markowitz</p>', 
            unsafe_allow_html=True)

# ==================== SIDEBAR ====================

with st.sidebar:
    st.title("⚙️ Configurações")
    
    st.divider()
    
    st.subheader("📅 Período")
    data_inicio = st.date_input(
        "De:",
        value=datetime.now() - timedelta(days=365),
        max_value=datetime.now()
    )
    data_fim = st.date_input(
        "Até:",
        value=datetime.now(),
        max_value=datetime.now()
    )
    
    st.divider()
    
    st.subheader("💰 Capital")
    capital_inicial = st.number_input(
        "Capital Inicial (R$)",
        min_value=1000.0,
        value=10000.0,
        step=1000.0,
        format="%.2f"
    )
    
    st.divider()
    
    taxa_livre_risco = st.slider(
        "Taxa Livre de Risco (%/ano)",
        min_value=0.0,
        max_value=20.0,
        value=13.75,
        step=0.25
    ) / 100
    
    incluir_dividendos = st.checkbox("📈 Análise de Dividendos", value=True)

# ==================== SELEÇÃO DE ATIVOS ====================

st.header("🎯 Seleção de Ativos")

tab1, tab2, tab3 = st.tabs(["📁 Por Segmento", "⭐ Populares", "✍️ Manual"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        tipo_ativo = st.selectbox(
            "Tipo de Ativo",
            ["Ações", "FIIs", "ETFs", "BDRs"]
        )
        
        tipo_map = {
            "Ações": ativos_b3.ACOES_B3,
            "FIIs": ativos_b3.FIIS,
            "ETFs": ativos_b3.ETFS,
            "BDRs": ativos_b3.BDRS
        }
        
        dados_tipo = tipo_map[tipo_ativo]
        
        segmentos = st.multiselect(
            "Segmentos",
            options=list(dados_tipo.keys())
        )
    
    with col2:
        if segmentos:
            # Coletar TODOS os ativos dos segmentos selecionados
            ativos_disponiveis = []
            for seg in segmentos:
                ativos_disponiveis.extend(dados_tipo[seg])
            ativos_disponiveis = sorted(list(set(ativos_disponiveis)))
            
            st.info(f"📊 {len(ativos_disponiveis)} ativos disponíveis nos segmentos selecionados")
            
            # Mostrar os ativos em formato de tags
            st.write("**Ativos disponíveis:**")
            ativos_text = ", ".join(ativos_disponiveis)
            st.text_area("", value=ativos_text, height=100, disabled=True, label_visibility="collapsed")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("✅ Selecionar Todos", use_container_width=True):
                    st.session_state.ativos_finais = ativos_disponiveis
                    st.success(f"✅ {len(ativos_disponiveis)} ativos selecionados!")
                    st.rerun()
            
            with col_b:
                if st.button("🗑️ Limpar", use_container_width=True):
                    st.session_state.ativos_finais = []
                    st.rerun()
            
            with col_c:
                if st.button("🎲 10 Aleatórios", use_container_width=True):
                    import random
                    st.session_state.ativos_finais = random.sample(
                        ativos_disponiveis, 
                        min(10, len(ativos_disponiveis))
                    )
                    st.success(f"✅ {len(st.session_state.ativos_finais)} ativos selecionados!")
                    st.rerun()
        else:
            st.info("👈 Selecione um ou mais segmentos")

with tab2:
    populares = [
        'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'WEGE3.SA',
        'RENT3.SA', 'ABEV3.SA', 'B3SA3.SA', 'MGLU3.SA', 'RADL3.SA',
        'BBAS3.SA', 'EGIE3.SA', 'CPLE6.SA', 'RAIL3.SA', 'SUZB3.SA'
    ]
    
    selected = st.multiselect(
        "Ações Populares",
        options=populares,
        default=[]
    )
    
    if selected:
        st.session_state.ativos_finais = selected
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Selecionar Todos", key="todos_pop", use_container_width=True):
            st.session_state.ativos_finais = populares
            st.rerun()
    
    with col2:
        if st.button("🗑️ Limpar", key="limpar_pop", use_container_width=True):
            st.session_state.ativos_finais = []
            st.rerun()

with tab3:
    st.info("💡 Formato: PETR4.SA (Brasil) ou AAPL (EUA)")
    
    manual_input = st.text_area(
        "Digite os tickers",
        height=100,
        placeholder="PETR4.SA, VALE3.SA, ITUB4.SA"
    )
    
    auto_sa = st.checkbox("Adicionar .SA automaticamente")
    
    if manual_input:
        tickers = manual_input.replace(',', ' ').replace('\n', ' ').replace(';', ' ').split()
        tickers = [t.strip().upper() for t in tickers if t.strip()]
        
        if auto_sa:
            tickers = [t if '.' in t else f"{t}.SA" for t in tickers]
        
        tickers = list(set(tickers))
        
        st.session_state.ativos_finais = tickers
        st.success(f"✅ {len(tickers)} ativos adicionados")

# ==================== RESUMO ====================

st.divider()

ativos_finais = st.session_state.ativos_finais

if ativos_finais:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("🎯 Total", len(ativos_finais))
    
    with col2:
        with st.expander("📋 Ver Lista", expanded=False):
            cols = st.columns(5)
            for i, ativo in enumerate(sorted(ativos_finais)):
                with cols[i % 5]:
                    st.write(f"✓ {ativo}")
else:
    st.warning("⚠️ Selecione ativos para continuar")
    st.stop()

if data_inicio >= data_fim:
    st.error("❌ Data inicial deve ser anterior à final!")
    st.stop()

# ==================== ANÁLISE ====================

st.divider()

if st.button("🚀 INICIAR ANÁLISE COMPLETA", type="primary", use_container_width=True):
    
    # Download de dados
    with st.spinner("📥 Baixando dados..."):
        dados_dict, sucessos, erros = baixar_dados_ativos(ativos_finais, data_inicio, data_fim)
        
        if not dados_dict:
            st.error("❌ Não foi possível baixar dados!")
            st.stop()
        
        dados = pd.DataFrame(dados_dict)
        dados = dados.ffill().bfill()
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ Sucesso: {len(sucessos)} ativos")
        with col2:
            if erros:
                st.warning(f"⚠️ Erros: {len(erros)} ativos")
    
    ativos_com_dados = dados.columns.tolist()
    retornos = dados.pct_change().dropna()
    
    # Dividendos
    dividendos_dict = {}
    df_dividendos = None
    
    if incluir_dividendos:
        with st.spinner("💰 Coletando dividendos..."):
            dividendos_dict = baixar_dividendos(ativos_com_dados, data_inicio, data_fim)
            
            if dividendos_dict:
                df_dividendos = pd.DataFrame()
                for ativo, divs in dividendos_dict.items():
                    df_dividendos[ativo] = divs.resample('M').sum()
                df_dividendos = df_dividendos.fillna(0)
    
    # Otimização
    st.header("🎯 Otimização de Carteira")
    
    with st.spinner("🔄 Otimizando..."):
        retorno_esperado = retornos.mean()
        matriz_cov = retornos.cov()
        
        # 1. Máximo Sharpe
        pesos_sharpe = otimizar_sharpe(retorno_esperado, matriz_cov, taxa_livre_risco)
        ret_sharpe, vol_sharpe, sharpe_sharpe = calcular_metricas_portfolio(
            pesos_sharpe, retorno_esperado, matriz_cov, taxa_livre_risco
        )
        
        # 2. Mínima Volatilidade
        pesos_min_vol = otimizar_minima_volatilidade(matriz_cov)
        ret_min_vol, vol_min_vol, sharpe_min_vol = calcular_metricas_portfolio(
            pesos_min_vol, retorno_esperado, matriz_cov, taxa_livre_risco
        )
        
        # 3. Foco em Dividendos
        pesos_div = None
        ret_div = vol_div = sharpe_div = 0
        
        if df_dividendos is not None and not df_dividendos.empty:
            total_div = df_dividendos.sum()
            if total_div.sum() > 0:
                pesos_div = (total_div / total_div.sum()).values
                ret_div, vol_div, sharpe_div = calcular_metricas_portfolio(
                    pesos_div, retorno_esperado, matriz_cov, taxa_livre_risco
                )
    
    # Comparação
    st.subheader("📊 Comparação das Estratégias")
    
    estrategias = []
    
    estrategias.append({
        'Nome': '🏆 Máximo Sharpe',
        'Retorno': ret_sharpe * 100,
        'Volatilidade': vol_sharpe * 100,
        'Sharpe': sharpe_sharpe,
        'Pesos': pesos_sharpe
    })
    
    estrategias.append({
        'Nome': '🛡️ Mínima Volatilidade',
        'Retorno': ret_min_vol * 100,
        'Volatilidade': vol_min_vol * 100,
        'Sharpe': sharpe_min_vol,
        'Pesos': pesos_min_vol
    })
    
    if pesos_div is not None:
        estrategias.append({
            'Nome': '💰 Foco Dividendos',
            'Retorno': ret_div * 100,
            'Volatilidade': vol_div * 100,
            'Sharpe': sharpe_div,
            'Pesos': pesos_div
        })
    
    # Tabela
    df_comp = pd.DataFrame([{
        'Estratégia': e['Nome'],
        'Retorno (%)': f"{e['Retorno']:.2f}%",
        'Volatilidade (%)': f"{e['Volatilidade']:.2f}%",
        'Sharpe': f"{e['Sharpe']:.2f}"
    } for e in estrategias])
    
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    # Recomendação
    st.subheader("🎖️ Melhor Estratégia")
    
    melhor = max(estrategias, key=lambda x: x['Sharpe'])
    
    st.success(f"""
    **Recomendação:** {melhor['Nome']}
    
    - **Sharpe Ratio:** {melhor['Sharpe']:.2f} (o melhor!)
    - **Retorno Esperado:** {melhor['Retorno']:.2f}%
    - **Volatilidade:** {melhor['Volatilidade']:.2f}%
    """)
    
    # Detalhamento
    st.divider()
    st.header("📋 Detalhamento das Carteiras")
    
    for estrategia in estrategias:
        with st.expander(f"{estrategia['Nome']} - Detalhes"):
            df_alocacao = pd.DataFrame({
                'Ativo': ativos_com_dados,
                'Peso (%)': estrategia['Pesos'] * 100,
                'Valor (R$)': estrategia['Pesos'] * capital_inicial
            })
            df_alocacao = df_alocacao[df_alocacao['Peso (%)'] > 0.5].sort_values('Peso (%)', ascending=False)
            
            st.dataframe(
                df_alocacao.style.format({
                    'Peso (%)': '{:.2f}%',
                    'Valor (R$)': 'R$ {:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Download
            csv = df_alocacao.to_csv(index=False).encode('utf-8')
            st.download_button(
                f"📥 Download {estrategia['Nome']}",
                csv,
                f"carteira_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    # Dividendos
    if df_dividendos is not None and not df_dividendos.empty:
        st.divider()
        st.header("💰 Análise de Dividendos")
        
        fig_div = go.Figure()
        
        for ativo in df_dividendos.columns:
            fig_div.add_trace(go.Bar(
                name=ativo,
                x=df_dividendos.index.strftime('%b/%y'),
                y=df_dividendos[ativo]
            ))
        
        fig_div.update_layout(
            title="Dividendos Mensais",
            xaxis_title="Mês",
            yaxis_title="R$",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig_div, use_container_width=True)
        
        total = df_dividendos.sum().sum()
        media = df_dividendos.sum(axis=1).mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Total", f"R$ {total:,.2f}")
        with col2:
            st.metric("📅 Média Mensal", f"R$ {media:,.2f}")
        with col3:
            st.metric("📈 Projeção Anual", f"R$ {media * 12:,.2f}")

st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 Otimizador de Carteira - Markowitz | Yahoo Finance<br>
    ⚠️ Apenas para fins educacionais</p>
</div>
""", unsafe_allow_html=True)
