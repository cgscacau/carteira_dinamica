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
                    status.text(f"✅ {ticker}: {len(divs)} pagamentos encontrados")
        except Exception as e:
            status.text(f"⚠️ {ticker}: Erro ao coletar")
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
                showscale=True,
                colorbar=dict(title="Retorno (%)")
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
                showscale=True,
                colorbar=dict(title="Vol (%)")
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
        textfont={"size": 10},
        colorbar=dict(title="Correlação")
    ))
    
    fig_corr.update_layout(
        height=600,
        xaxis={'side': 'bottom'}
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # ========== EVOLUÇÃO DOS PREÇOS ==========
    st.subheader("📈 Evolução Histórica dos Preços")
    
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
    
    # ========== OTIMIZAÇÃO ==========
    st.divider()
    st.header("🎯 Otimização de Carteira")
    
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
        
    # ========== COLETA DE DIVIDENDOS MELHORADA ==========
    dividendos_dict = {}
    df_dividendos = None
    
    if incluir_dividendos:
        st.divider()
        st.header("💰 Análise de Dividendos")
        
        with st.spinner("💰 Coletando dados de dividendos..."):
            dividendos_dict = baixar_dividendos(ativos_com_dados, data_inicio, data_fim)
            
            if dividendos_dict:
                st.success(f"✅ Dividendos encontrados para {len(dividendos_dict)} ativos")
                
                # Criar DataFrame mensal
                df_dividendos = pd.DataFrame()
                
                for ativo, divs in dividendos_dict.items():
                    # Agrupar por mês
                    divs_mensais = divs.resample('M').sum()
                    df_dividendos[ativo] = divs_mensais
                
                # Preencher valores faltantes com 0
                df_dividendos = df_dividendos.fillna(0)
                
                # Remover meses sem nenhum dividendo
                df_dividendos = df_dividendos[df_dividendos.sum(axis=1) > 0]
                
                if not df_dividendos.empty:
                    # ========== RESUMO RÁPIDO ==========
                    total_divs = df_dividendos.sum().sum()
                    ativos_pagantes = len(dividendos_dict)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("💰 Total de Dividendos", f"R$ {total_divs:,.2f}")
                    with col2:
                        st.metric("📊 Ativos Pagantes", f"{ativos_pagantes}/{len(ativos_com_dados)}")
                    
                    # ========== GRÁFICO DE BARRAS EMPILHADAS ==========
                    st.subheader("📊 Distribuição Mensal de Dividendos")
                    
                    fig_div = go.Figure()
                    
                    for ativo in df_dividendos.columns:
                        fig_div.add_trace(go.Bar(
                            name=ativo,
                            x=df_dividendos.index.strftime('%b/%y'),
                            y=df_dividendos[ativo],
                            text=df_dividendos[ativo].apply(lambda x: f'R$ {x:.2f}' if x > 0.01 else ''),
                            textposition='inside',
                            textfont=dict(size=9),
                            hovertemplate='<b>%{fullData.name}</b><br>R$ %{y:.2f}<extra></extra>'
                        ))
                    
                    fig_div.update_layout(
                        xaxis_title="Mês",
                        yaxis_title="Dividendos (R$)",
                        barmode='stack',
                        height=500,
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    
                    st.plotly_chart(fig_div, use_container_width=True)
                    
                    # ========== TABELA DE DIVIDENDOS MENSAIS ==========
                    st.subheader("📅 Tabela Detalhada de Dividendos Mensais")
                    
                    # Preparar tabela
                    df_div_display = df_dividendos.copy()
                    df_div_display.index = df_div_display.index.strftime('%b/%Y')
                    
                    # Adicionar coluna de total mensal
                    df_div_display['💰 TOTAL MÊS'] = df_div_display.sum(axis=1)
                    
                    # Adicionar linha de total por ativo
                    totais = df_div_display.sum()
                    df_div_display.loc['🏆 TOTAL GERAL'] = totais
                    
                    # Formatar e exibir
                    st.dataframe(
                        df_div_display.style.format('R$ {:.2f}').background_gradient(
                            cmap='Greens',
                            axis=None
                        ).set_properties(**{
                            'font-weight': 'bold',
                            'background-color': '#90EE90'
                        }, subset=pd.IndexSlice['🏆 TOTAL GERAL', :]).set_properties(**{
                            'font-weight': 'bold',
                            'background-color': '#98FB98'
                        }, subset=pd.IndexSlice[:, '💰 TOTAL MÊS']),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Botão para download da tabela
                    csv_dividendos = df_div_display.to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Download Tabela de Dividendos (CSV)",
                        data=csv_dividendos,
                        file_name=f"dividendos_mensais_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="download_dividendos_tabela"
                    )
                    
                    # ========== GRÁFICO DE DIVIDENDOS ACUMULADOS ==========
                    st.subheader("📈 Evolução dos Dividendos Acumulados")
                    
                    df_div_acum = df_dividendos.cumsum()
                    
                    fig_div_acum = go.Figure()
                    
                    for ativo in df_div_acum.columns:
                        fig_div_acum.add_trace(go.Scatter(
                            name=ativo,
                            x=df_div_acum.index.strftime('%b/%y'),
                            y=df_div_acum[ativo],
                            mode='lines+markers',
                            line=dict(width=2),
                            marker=dict(size=6),
                            hovertemplate='<b>%{fullData.name}</b><br>Acumulado: R$ %{y:.2f}<extra></extra>'
                        ))
                    
                    fig_div_acum.update_layout(
                        xaxis_title="Mês",
                        yaxis_title="Dividendos Acumulados (R$)",
                        height=450,
                        hovermode='x unified',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_div_acum, use_container_width=True)
                    
                    # ========== MÉTRICAS GERAIS ==========
                    st.subheader("📊 Métricas Gerais de Dividendos")
                    
                    total = df_dividendos.sum().sum()
                    media_mensal = df_dividendos.sum(axis=1).mean()
                    mediana_mensal = df_dividendos.sum(axis=1).median()
                    projecao = media_mensal * 12
                    meses_pagantes = len(df_dividendos)
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric(
                            "💰 Total Recebido",
                            f"R$ {total:,.2f}",
                            help="Total de dividendos no período"
                        )
                    
                    with col2:
                        st.metric(
                            "📅 Média Mensal",
                            f"R$ {media_mensal:,.2f}",
                            help="Média de dividendos por mês"
                        )
                    
                    with col3:
                        st.metric(
                            "📊 Mediana Mensal",
                            f"R$ {mediana_mensal:,.2f}",
                            help="Valor mediano mensal"
                        )
                    
                    with col4:
                        st.metric(
                            "📈 Projeção Anual",
                            f"R$ {projecao:,.2f}",
                            help="Projeção baseada na média"
                        )
                    
                    with col5:
                        st.metric(
                            "✅ Meses com Pagamento",
                            f"{meses_pagantes}",
                            help="Meses que receberam dividendos"
                        )
                    
                    # ========== ANÁLISE POR ATIVO ==========
                    st.subheader("🏆 Análise Detalhada por Ativo")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**💰 Ranking de Dividendos Totais**")
                        
                        div_totais = df_dividendos.sum().sort_values(ascending=False)
                        
                        fig_rank = go.Figure(go.Bar(
                            x=div_totais.values,
                            y=div_totais.index,
                            orientation='h',
                            marker=dict(
                                color=div_totais.values,
                                colorscale='Greens',
                                showscale=True,
                                colorbar=dict(title="R$")
                            ),
                            text=div_totais.apply(lambda x: f'R$ {x:.2f}'),
                            textposition='outside',
                            hovertemplate='<b>%{y}</b><br>Total: R$ %{x:.2f}<extra></extra>'
                        ))
                        
                        fig_rank.update_layout(
                            xaxis_title="Dividendos Totais (R$)",
                            yaxis_title="Ativo",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_rank, use_container_width=True)
                        
                        # Tabela de totais
                        df_totais = pd.DataFrame({
                            'Ativo': div_totais.index,
                            'Total (R$)': div_totais.values,
                            '% do Total': (div_totais.values / div_totais.sum() * 100),
                            'Média Mensal': div_totais.values / meses_pagantes
                        })
                        
                        st.dataframe(
                            df_totais.style.format({
                                'Total (R$)': 'R$ {:.2f}',
                                '% do Total': '{:.1f}%',
                                'Média Mensal': 'R$ {:.2f}'
                            }).background_gradient(subset=['Total (R$)'], cmap='Greens'),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col2:
                        st.write("**📊 Dividend Yield Anualizado**")
                        
                        dy_data = []
                        for ativo in df_dividendos.columns:
                            if ativo in dados.columns:
                                preco_medio = dados[ativo].mean()
                                div_total = df_dividendos[ativo].sum()
                                div_anual = (div_total / meses_pagantes) * 12
                                dy = (div_anual / preco_medio) * 100 if preco_medio > 0 else 0
                                
                                dy_data.append({
                                    'Ativo': ativo,
                                    'DY (%)': dy,
                                    'Div Anual': div_anual,
                                    'Preço Médio': preco_medio
                                })
                        
                        df_dy = pd.DataFrame(dy_data).sort_values('DY (%)', ascending=False)
                        
                        fig_dy = go.Figure(go.Bar(
                            x=df_dy['DY (%)'],
                            y=df_dy['Ativo'],
                            orientation='h',
                            marker=dict(
                                color=df_dy['DY (%)'],
                                colorscale='YlGn',
                                showscale=True,
                                colorbar=dict(title="DY %")
                            ),
                            text=df_dy['DY (%)'].apply(lambda x: f'{x:.2f}%'),
                            textposition='outside',
                            hovertemplate='<b>%{y}</b><br>DY: %{x:.2f}%<extra></extra>'
                        ))
                        
                        fig_dy.update_layout(
                            xaxis_title="Dividend Yield Anual (%)",
                            yaxis_title="Ativo",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_dy, use_container_width=True)
                        
                        # Tabela de DY
                        st.dataframe(
                            df_dy[['Ativo', 'DY (%)', 'Div Anual']].style.format({
                                'DY (%)': '{:.2f}%',
                                'Div Anual': 'R$ {:.2f}'
                            }).background_gradient(subset=['DY (%)'], cmap='YlGn'),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # ========== ANÁLISE DE FREQUÊNCIA ==========
                    st.subheader("📅 Frequência de Pagamentos")
                    
                    freq_pagamentos = (df_dividendos > 0).sum().sort_values(ascending=False)
                    
                    fig_freq = go.Figure(go.Bar(
                        x=freq_pagamentos.index,
                        y=freq_pagamentos.values,
                        marker=dict(
                            color=freq_pagamentos.values,
                            colorscale='Blues',
                            showscale=True
                        ),
                        text=freq_pagamentos.apply(lambda x: f'{x}/{meses_pagantes}'),
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Pagou em %{y} meses<extra></extra>'
                    ))
                    
                    fig_freq.update_layout(
                        title="Quantidade de Meses com Pagamento",
                        xaxis_title="Ativo",
                        yaxis_title="Meses com Pagamento",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_freq, use_container_width=True)
                    
                    # ========== COMPARAÇÃO COM CARTEIRAS ==========
                    st.subheader("💼 Dividendos Estimados por Estratégia")
                    
                    if len(estrategias) > 0:
                        div_estrategias = []
                        
                        for estrategia in estrategias:
                            div_anual_estimado = 0
                            
                            for i, ativo in enumerate(ativos_com_dados):
                                if ativo in df_dividendos.columns:
                                    peso = estrategia['Pesos'][i]
                                    div_ativo_anual = (df_dividendos[ativo].sum() / meses_pagantes) * 12
                                    div_anual_estimado += peso * div_ativo_anual * capital_inicial
                            
                            div_estrategias.append({
                                'Estratégia': estrategia['Nome'],
                                'Dividendos Anuais': div_anual_estimado,
                                'Dividendos Mensais': div_anual_estimado / 12,
                                'DY (%)': (div_anual_estimado / capital_inicial) * 100
                            })
                        
                        df_div_estrategias = pd.DataFrame(div_estrategias)
                        
                        # Gráfico
                        fig_comp_div = go.Figure()
                        
                        fig_comp_div.add_trace(go.Bar(
                            x=df_div_estrategias['Estratégia'],
                            y=df_div_estrategias['Dividendos Anuais'],
                            marker=dict(color=['#FF4B4B', '#00CC00', '#FFD700'][:len(df_div_estrategias)]),
                            text=df_div_estrategias['Dividendos Anuais'].apply(lambda x: f'R$ {x:,.2f}'),
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Anual: R$ %{y:,.2f}<extra></extra>'
                        ))
                        
                        fig_comp_div.update_layout(
                            title="Dividendos Anuais Estimados",
                            xaxis_title="Estratégia",
                            yaxis_title="Dividendos (R$)",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_comp_div, use_container_width=True)
                        
                        # Tabela
                        st.dataframe(
                            df_div_estrategias.style.format({
                                'Dividendos Anuais': 'R$ {:.2f}',
                                'Dividendos Mensais': 'R$ {:.2f}',
                                'DY (%)': '{:.2f}%'
                            }).background_gradient(subset=['Dividendos Anuais'], cmap='Greens'),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Destaque
                        melhor_div = df_div_estrategias.loc[df_div_estrategias['Dividendos Anuais'].idxmax()]
                        
                        st.success(f"""
                        **🏆 Melhor Estratégia para Renda Passiva:** {melhor_div['Estratégia']}
                        
                        - 💰 Dividendos Anuais: R$ {melhor_div['Dividendos Anuais']:,.2f}
                        - 📅 Dividendos Mensais: R$ {melhor_div['Dividendos Mensais']:,.2f}
                        - 📊 Dividend Yield: {melhor_div['DY (%)']:.2f}%
                        """)
                
                else:
                    st.warning("⚠️ Dividendos encontrados mas todos os valores são zero no período.")
            
            else:
                st.warning(f"""
                ⚠️ **Nenhum dividendo encontrado** para os {len(ativos_com_dados)} ativos no período de {data_inicio.strftime('%d/%m/%Y')} a {data_fim.strftime('%d/%m/%Y')}.
                
                **Ativos analisados:** {', '.join(ativos_com_dados)}
                
                **Possíveis causas:**
                1. Os ativos não pagaram dividendos no período
                2. Yahoo Finance não tem os dados disponíveis
                3. Período muito curto
                
                **Sugestão:** Tente:
                - Ampliar o período de análise
                - Selecionar FIIs conhecidos: HGLG11.SA, MXRF11.SA, KNRI11.SA
                - Selecionar ações: ITUB4.SA, BBDC4.SA, TAEE11.SA
                """)

        
        # ========== COMPARAÇÃO COM CARTEIRAS ==========
        st.subheader("💼 Dividendos Estimados por Estratégia de Carteira")
        
        if len(estrategias) > 0:
            div_estrategias = []
            
            for estrategia in estrategias:
                # Calcular dividendos anuais estimados para cada estratégia
                div_anual_estimado = 0
                
                for i, ativo in enumerate(ativos_com_dados):
                    if ativo in df_dividendos.columns:
                        peso = estrategia['Pesos'][i]
                        div_ativo = (df_dividendos[ativo].sum() / len(df_dividendos)) * 12
                        div_anual_estimado += peso * div_ativo * capital_inicial
                
                div_estrategias.append({
                    'Estratégia': estrategia['Nome'],
                    'Dividendos Anuais': div_anual_estimado,
                    'Dividendos Mensais': div_anual_estimado / 12,
                    'DY (%)': (div_anual_estimado / capital_inicial) * 100
                })
            
            df_div_estrategias = pd.DataFrame(div_estrategias)
            
            # Gráfico de comparação
            fig_comp_div = go.Figure()
            
            fig_comp_div.add_trace(go.Bar(
                x=df_div_estrategias['Estratégia'],
                y=df_div_estrategias['Dividendos Anuais'],
                marker=dict(color=['#FF4B4B', '#00CC00', '#FFD700'][:len(df_div_estrategias)]),
                text=df_div_estrategias['Dividendos Anuais'].apply(lambda x: f'R$ {x:,.2f}'),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Anual: R$ %{y:,.2f}<extra></extra>'
            ))
            
            fig_comp_div.update_layout(
                title="Dividendos Anuais Estimados por Estratégia",
                xaxis_title="Estratégia",
                yaxis_title="Dividendos Anuais (R$)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_comp_div, use_container_width=True)
            
            # Tabela de comparação
            st.dataframe(
                df_div_estrategias.style.format({
                    'Dividendos Anuais': 'R$ {:.2f}',
                    'Dividendos Mensais': 'R$ {:.2f}',
                    'DY (%)': '{:.2f}%'
                }).background_gradient(subset=['Dividendos Anuais'], cmap='Greens'),
                use_container_width=True,
                hide_index=True
            )
            
            # Destaque para melhor dividendo
            melhor_div = df_div_estrategias.loc[df_div_estrategias['Dividendos Anuais'].idxmax()]
            
            st.success(f"""
            **🏆 Melhor Estratégia para Dividendos:** {melhor_div['Estratégia']}
            
            - Dividendos Anuais: R$ {melhor_div['Dividendos Anuais']:,.2f}
            - Dividendos Mensais: R$ {melhor_div['Dividendos Mensais']:,.2f}
            - Dividend Yield: {melhor_div['DY (%)']:.2f}%
            """)
        
        # ========== CALENDÁRIO DE DIVIDENDOS ==========
        st.subheader("📆 Calendário de Pagamentos")
        
        st.info("""
        **💡 Dica:** Os dividendos mostrados são baseados no histórico do período selecionado. 
        Para uma carteira focada em renda passiva, considere a estratégia "💰 Foco Dividendos" 
        que aloca maior peso nos ativos que mais pagam dividendos.
        """)
    else:
        st.info("""
        ℹ️ **Nenhum dividendo encontrado** no período selecionado para os ativos escolhidos.
        
        **Possíveis razões:**
        - Os ativos selecionados não pagaram dividendos no período
        - O período de análise é muito curto
        - Os dados de dividendos não estão disponíveis no Yahoo Finance
        
        **Sugestão:** Tente selecionar ativos conhecidos por pagar dividendos regulares, como:
        - ITUB4.SA, BBDC4.SA (Bancos)
        - TAEE11.SA, CPLE6.SA (Energia)
        - FIIs (Fundos Imobiliários)
        """)

    # Comparação
    st.subheader("📊 Comparação das Estratégias")
    
    estrategias = []
    cores = ['#FF4B4B', '#00CC00', '#FFD700']
    
    estrategias.append({
        'Nome': '🏆 Máximo Sharpe',
        'Retorno': ret_sharpe * 100,
        'Volatilidade': vol_sharpe * 100,
        'Sharpe': sharpe_sharpe,
        'Pesos': pesos_sharpe,
        'Cor': cores[0]
    })
    
    estrategias.append({
        'Nome': '🛡️ Mínima Volatilidade',
        'Retorno': ret_min_vol * 100,
        'Volatilidade': vol_min_vol * 100,
        'Sharpe': sharpe_min_vol,
        'Pesos': pesos_min_vol,
        'Cor': cores[1]
    })
    
    if pesos_div is not None:
        estrategias.append({
            'Nome': '💰 Foco Dividendos',
            'Retorno': ret_div * 100,
            'Volatilidade': vol_div * 100,
            'Sharpe': sharpe_div,
            'Pesos': pesos_div,
            'Cor': cores[2]
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
    melhor = max(estrategias, key=lambda x: x['Sharpe'])
    
    st.success(f"""
    **🎖️ Melhor Estratégia:** {melhor['Nome']}
    
    - Sharpe: {melhor['Sharpe']:.2f}
    - Retorno: {melhor['Retorno']:.2f}%
    - Volatilidade: {melhor['Volatilidade']:.2f}%
    """)
    
    # Detalhamento
    st.divider()
    st.header("📋 Detalhamento das Carteiras")
    
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
            nome_arquivo = estrategia['Nome'].replace('🏆 ', '').replace('🛡️ ', '').replace('💰 ', '').replace(' ', '_').lower()
            
            st.download_button(
                label=f"📥 Download {estrategia['Nome']}",
                data=csv,
                file_name=f"{nome_arquivo}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"download_estrategia_{idx}"
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
    <p>📊 Otimizador de Carteira - Markowitz<br>
    ⚠️ Apenas para fins educacionais</p>
</div>
""", unsafe_allow_html=True)
