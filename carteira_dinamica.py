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
    """Baixa dados de dividendos com múltiplas tentativas"""
    dividendos_dict = {}
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    for i, ticker in enumerate(tickers):
        status.text(f"💰 Coletando dividendos de {ticker}...")
        
        try:
            ativo = yf.Ticker(ticker)
            
            # Método 1: Tentar pegar dividendos diretamente
            divs = ativo.dividends
            
            # Se não conseguir, tentar pelo histórico
            if divs.empty:
                hist = ativo.history(start=data_inicio, end=data_fim)
                if 'Dividends' in hist.columns:
                    divs = hist['Dividends']
                    divs = divs[divs > 0]
            
            if not divs.empty:
                # Filtrar pelo período
                divs = divs[(divs.index >= pd.Timestamp(data_inicio)) & 
                           (divs.index <= pd.Timestamp(data_fim))]
                
                if not divs.empty and len(divs) > 0:
                    dividendos_dict[ticker] = divs
                    status.text(f"✅ {ticker}: {len(divs)} pagamentos")
        except:
            pass
        
        progress_bar.progress((i + 1) / len(tickers))
    
    progress_bar.empty()
    status.empty()
    
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
            ativos_disponiveis = []
            for seg in segmentos:
                ativos_disponiveis.extend(dados_tipo[seg])
            ativos_disponiveis = sorted(list(set(ativos_disponiveis)))
            
            st.info(f"📊 {len(ativos_disponiveis)} ativos disponíveis")
            
            st.write("**Ativos disponíveis:**")
            ativos_text = ", ".join(ativos_disponiveis)
            st.text_area("", value=ativos_text, height=100, disabled=True, label_visibility="collapsed")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("✅ Todos", use_container_width=True):
                    st.session_state.ativos_finais = ativos_disponiveis
                    st.success(f"✅ {len(ativos_disponiveis)} ativos!")
                    st.rerun()
            
            with col_b:
                if st.button("🗑️ Limpar", use_container_width=True):
                    st.session_state.ativos_finais = []
                    st.rerun()
            
            with col_c:
                if st.button("🎲 10 Aleat.", use_container_width=True):
                    import random
                    st.session_state.ativos_finais = random.sample(
                        ativos_disponiveis, 
                        min(10, len(ativos_disponiveis))
                    )
                    st.success(f"✅ {len(st.session_state.ativos_finais)} ativos!")
                    st.rerun()
        else:
            st.info("👈 Selecione segmentos")

with tab2:
    populares = [
        'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'WEGE3.SA',
        'RENT3.SA', 'ABEV3.SA', 'B3SA3.SA', 'MGLU3.SA', 'RADL3.SA'
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
        if st.button("✅ Todos", key="todos_pop", use_container_width=True):
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
        st.success(f"✅ {len(tickers)} ativos")

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
                with st.expander(f"⚠️ Erros: {len(erros)} ativos"):
                    for erro in erros:
                        st.write(f"• {erro}")
    
    ativos_com_dados = dados.columns.tolist()
    retornos = dados.pct_change().dropna()
    
    # ========== ESTATÍSTICAS BÁSICAS ==========
    st.header("📊 Estatísticas dos Ativos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Retorno Médio Diário")
        retorno_medio = (retornos.mean() * 100).sort_values(ascending=False)
        
        fig_ret = go.Figure(go.Bar(
            x=retorno_medio.values,
            y=retorno_medio.index,
            orientation='h',
            marker=dict(
                color=retorno_medio.values,
                colorscale='RdYlGn',
                showscale=True
            ),
            text=retorno_medio.apply(lambda x: f'{x:.3f}%'),
            textposition='outside'
        ))
        
        fig_ret.update_layout(
            xaxis_title="Retorno (%)",
            yaxis_title="Ativo",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_ret, use_container_width=True)
    
    with col2:
        st.subheader("📊 Volatilidade")
        volatilidade = (retornos.std() * 100 * np.sqrt(252)).sort_values(ascending=False)
        
        fig_vol = go.Figure(go.Bar(
            x=volatilidade.values,
            y=volatilidade.index,
            orientation='h',
            marker=dict(
                color=volatilidade.values,
                colorscale='Reds',
                showscale=True
            ),
            text=volatilidade.apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ))
        
        fig_vol.update_layout(
            xaxis_title="Volatilidade (%)",
            yaxis_title="Ativo",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    # ========== MATRIZ DE CORRELAÇÃO ==========
    st.subheader("🔗 Matriz de Correlação")
    
    correlacao = retornos.corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlacao.values,
        x=correlacao.columns,
        y=correlacao.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlacao.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig_corr.update_layout(height=600)
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # ========== EVOLUÇÃO DOS PREÇOS ==========
    st.subheader("📈 Evolução Histórica")
    
    dados_norm = (dados / dados.iloc[0] * 100)
    
    fig_precos = go.Figure()
    
    for ativo in dados_norm.columns:
        fig_precos.add_trace(go.Scatter(
            x=dados_norm.index,
            y=dados_norm[ativo],
            mode='lines',
            name=ativo
        ))
    
    fig_precos.update_layout(
        xaxis_title="Data",
        yaxis_title="Valor (Base 100)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_precos, use_container_width=True)
    
    # ========== DIVIDENDOS ==========
    dividendos_dict = {}
    df_dividendos = None
    
    if incluir_dividendos:
        st.divider()
        st.header("💰 Análise de Dividendos")
        
        with st.spinner("💰 Coletando dividendos..."):
            dividendos_dict = baixar_dividendos(ativos_com_dados, data_inicio, data_fim)
            
            if dividendos_dict:
                st.success(f"✅ Dividendos encontrados para {len(dividendos_dict)} ativos")
                
                df_dividendos = pd.DataFrame()
                
                for ativo, divs in dividendos_dict.items():
                    divs_mensais = divs.resample('M').sum()
                    df_dividendos[ativo] = divs_mensais
                
                df_dividendos = df_dividendos.fillna(0)
                df_dividendos = df_dividendos[df_dividendos.sum(axis=1) > 0]
                
                if not df_dividendos.empty:
                    total_divs = df_dividendos.sum().sum()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("💰 Total", f"R$ {total_divs:,.2f}")
                    with col2:
                        st.metric("📊 Ativos Pagantes", f"{len(dividendos_dict)}/{len(ativos_com_dados)}")
                    
                    # Gráfico
                    fig_div = go.Figure()
                    
                    for ativo in df_dividendos.columns:
                        fig_div.add_trace(go.Bar(
                            name=ativo,
                            x=df_dividendos.index.strftime('%b/%y'),
                            y=df_dividendos[ativo]
                        ))
                    
                    fig_div.update_layout(
                        xaxis_title="Mês",
                        yaxis_title="R$",
                        barmode='stack',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_div, use_container_width=True)
                    
                    # Tabela
                    st.subheader("📅 Tabela Mensal")
                    
                    df_div_display = df_dividendos.copy()
                    df_div_display.index = df_div_display.index.strftime('%b/%Y')
                    df_div_display['💰 TOTAL'] = df_div_display.sum(axis=1)
                    df_div_display.loc['🏆 TOTAL'] = df_div_display.sum()
                    
                    st.dataframe(
                        df_div_display.style.format('R$ {:.2f}').background_gradient(cmap='Greens'),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Métricas
                    media = df_dividendos.sum(axis=1).mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("💰 Total", f"R$ {total_divs:,.2f}")
                    with col2:
                        st.metric("📅 Média Mensal", f"R$ {media:,.2f}")
                    with col3:
                        st.metric("📈 Projeção Anual", f"R$ {media * 12:,.2f}")
            else:
                st.warning("⚠️ Nenhum dividendo encontrado no período")
    
    # ========== OTIMIZAÇÃO ==========
    st.divider()
    st.header("🎯 Otimização")
    
    with st.spinner("🔄 Otimizando..."):
        retorno_esperado = retornos.mean()
        matriz_cov = retornos.cov()
        
        pesos_sharpe = otimizar_sharpe(retorno_esperado, matriz_cov, taxa_livre_risco)
        ret_sharpe, vol_sharpe, sharpe_sharpe = calcular_metricas_portfolio(
            pesos_sharpe, retorno_esperado, matriz_cov, taxa_livre_risco
        )
        
        pesos_min_vol = otimizar_minima_volatilidade(matriz_cov)
        ret_min_vol, vol_min_vol, sharpe_min_vol = calcular_metricas_portfolio(
            pesos_min_vol, retorno_esperado, matriz_cov, taxa_livre_risco
        )
        
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
    st.subheader("📊 Comparação")
    
    estrategias = []
    
    estrategias.append({
        'Nome': '🏆 Máximo Sharpe',
        'Retorno': ret_sharpe * 100,
        'Volatilidade': vol_sharpe * 100,
        'Sharpe': sharpe_sharpe,
        'Pesos': pesos_sharpe,
        'Cor': '#FF4B4B'
    })
    
    estrategias.append({
        'Nome': '🛡️ Mínima Volatilidade',
        'Retorno': ret_min_vol * 100,
        'Volatilidade': vol_min_vol * 100,
        'Sharpe': sharpe_min_vol,
        'Pesos': pesos_min_vol,
        'Cor': '#00CC00'
    })
    
    if pesos_div is not None:
        estrategias.append({
            'Nome': '💰 Foco Dividendos',
            'Retorno': ret_div * 100,
            'Volatilidade': vol_div * 100,
            'Sharpe': sharpe_div,
            'Pesos': pesos_div,
            'Cor': '#FFD700'
        })
    
    # Tabela
    df_comp = pd.DataFrame([{
        'Estratégia': e['Nome'],
        'Retorno (%)': f"{e['Retorno']:.2f}%",
        'Volatilidade (%)': f"{e['Volatilidade']:.2f}%",
        'Sharpe': f"{e['Sharpe']:.2f}"
    } for e in estrategias])
    
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    melhor = max(estrategias, key=lambda x: x['Sharpe'])
    
    st.success(f"""
    **🎖️ Melhor:** {melhor['Nome']}
    
    - Sharpe: {melhor['Sharpe']:.2f}
    - Retorno: {melhor['Retorno']:.2f}%
    - Volatilidade: {melhor['Volatilidade']:.2f}%
    """)
    
    # Detalhamento
    st.divider()
    st.header("📋 Detalhamento")
    
    for idx, estrategia in enumerate(estrategias):
        with st.expander(f"{estrategia['Nome']}"):
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
            
            csv = df_alocacao.to_csv(index=False).encode('utf-8')
            st.download_button(
                f"📥 Download",
                csv,
                f"carteira_{idx}_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                key=f"download_{idx}"
            )
    
    # Fronteira Eficiente
    st.divider()
    st.header("📈 Fronteira Eficiente")
    
    with st.spinner("Calculando..."):
        n_portfolios = 5000
        resultados = np.zeros((3, n_portfolios))
        
        for i in range(n_portfolios):
            pesos = np.random.random(len(ativos_com_dados))
            pesos /= pesos.sum()
            
            ret, vol, sharpe = calcular_metricas_portfolio(
                pesos, retorno_esperado, matriz_cov, taxa_livre_risco
            )
            
            resultados[0, i] = ret
            resultados[1, i] = vol
            resultados[2, i] = sharpe
    
    fig_front = go.Figure()
    
    fig_front.add_trace(go.Scatter(
        x=resultados[1, :] * 100,
        y=resultados[0, :] * 100,
        mode='markers',
        marker=dict(
            size=5,
            color=resultados[2, :],
            colorscale='Viridis',
            showscale=True
        ),
        name='Carteiras'
    ))
    
    for estrategia in estrategias:
        fig_front.add_trace(go.Scatter(
            x=[estrategia['Volatilidade']],
            y=[estrategia['Retorno']],
            mode='markers',
            marker=dict(size=25, color=estrategia['Cor'], symbol='star'),
            name=estrategia['Nome']
        ))
    
    fig_front.update_layout(
        xaxis_title="Volatilidade (%)",
        yaxis_title="Retorno (%)",
        height=600
    )
    
    st.plotly_chart(fig_front, use_container_width=True)
    
    # Performance
    st.divider()
    st.header("📊 Performance Histórica")
    
    fig_perf = go.Figure()
    
    for estrategia in estrategias:
        retornos_estrategia = (retornos * estrategia['Pesos']).sum(axis=1)
        valor_estrategia = capital_inicial * (1 + retornos_estrategia).cumprod()
        
        fig_perf.add_trace(go.Scatter(
            x=valor_estrategia.index,
            y=valor_estrategia.values,
            mode='lines',
            name=estrategia['Nome'],
            line=dict(width=3, color=estrategia['Cor'])
        ))
    
    fig_perf.update_layout(
        xaxis_title="Data",
        yaxis_title="Valor (R$)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_perf, use_container_width=True)

st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 Otimizador de Carteira - Markowitz<br>
    ⚠️ Apenas para fins educacionais</p>
</div>
""", unsafe_allow_html=True)
