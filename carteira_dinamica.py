import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ativos_b3

# ==================== CONFIGURAÇÃO ====================
st.set_page_config(
    page_title="Otimizador de Carteira - Markowitz",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS CUSTOMIZADO ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== FUNÇÕES AUXILIARES ====================

def baixar_dados_ativos(tickers, data_inicio, data_fim):
    """Baixa dados de múltiplos ativos com tratamento de erro robusto"""
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
                erros.append(f"{ticker} (dados insuficientes)")
        except Exception as e:
            erros.append(f"{ticker} (erro)")
        
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

def calcular_pesos_dividendos(df_dividendos):
    """Calcula pesos baseados em dividendos"""
    total_div = df_dividendos.sum()
    if total_div.sum() > 0:
        return (total_div / total_div.sum()).values
    return None

# ==================== INICIALIZAÇÃO ====================

if 'ativos_selecionados' not in st.session_state:
    st.session_state.ativos_selecionados = []

# ==================== HEADER ====================

st.markdown('<p class="main-header">📊 Otimizador de Carteira - Teoria de Markowitz</p>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Análise Quantitativa para Maximização de Retorno Ajustado ao Risco</p>', 
            unsafe_allow_html=True)

# ==================== SIDEBAR ====================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/financial-growth-analysis.png", width=80)
    st.title("⚙️ Configurações")
    
    st.divider()
    
    st.subheader("📅 Período de Análise")
    col1, col2 = st.columns(2)
    with col1:
        data_inicio = st.date_input(
            "De:",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )
    with col2:
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
    
    st.subheader("📊 Parâmetros")
    taxa_livre_risco = st.slider(
        "Taxa Livre de Risco (%/ano)",
        min_value=0.0,
        max_value=20.0,
        value=13.75,
        step=0.25
    ) / 100
    
    incluir_dividendos = st.checkbox("📈 Incluir Análise de Dividendos", value=True)
    
    st.divider()
    
    st.info("💡 **Dica:** Selecione pelo menos 5 ativos para uma diversificação adequada.")

# ==================== SELEÇÃO DE ATIVOS ====================

st.header("🎯 Seleção de Ativos")

tab1, tab2, tab3 = st.tabs(["📁 Por Segmento", "⭐ Populares", "✍️ Manual"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        tipo_ativo = st.selectbox(
            "Tipo de Ativo",
            ["Ações", "FIIs", "ETFs", "BDRs"],
            key="tipo_ativo"
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
            options=list(dados_tipo.keys()),
            key="segmentos"
        )
    
    with col2:
        if segmentos:
            ativos_disponiveis = []
            for seg in segmentos:
                ativos_disponiveis.extend(dados_tipo[seg])
            ativos_disponiveis = sorted(list(set(ativos_disponiveis)))
            
            st.info(f"📊 {len(ativos_disponiveis)} ativos disponíveis")
            
            # Verificar se os ativos selecionados estão na lista disponível
            default_segmento = []
            if st.session_state.ativos_selecionados:
                default_segmento = [a for a in st.session_state.ativos_selecionados if a in ativos_disponiveis]
            
            if not default_segmento and ativos_disponiveis:
                default_segmento = []
            
            selected = st.multiselect(
                "Ativos",
                options=ativos_disponiveis,
                default=default_segmento,
                key="multi_segmento"
            )
            
            st.session_state.ativos_selecionados = selected
            
            col_a, col_b, col_c = st.columns(3)
            
            if col_a.button("✅ Todos", key="btn_todos", use_container_width=True):
                st.session_state.ativos_selecionados = ativos_disponiveis
                st.rerun()
            
            if col_b.button("🗑️ Limpar", key="btn_limpar", use_container_width=True):
                st.session_state.ativos_selecionados = []
                st.rerun()
            
            if col_c.button("🎲 10 Aleat.", key="btn_random", use_container_width=True):
                import random
                st.session_state.ativos_selecionados = random.sample(
                    ativos_disponiveis, 
                    min(10, len(ativos_disponiveis))
                )
                st.rerun()
        else:
            st.info("👈 Selecione segmentos à esquerda")

with tab2:
    populares = [
        'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'WEGE3.SA',
        'RENT3.SA', 'ABEV3.SA', 'B3SA3.SA', 'MGLU3.SA', 'RADL3.SA'
    ]
    
    # Verificar se os ativos selecionados estão na lista de populares
    default_populares = []
    if st.session_state.ativos_selecionados:
        default_populares = [a for a in st.session_state.ativos_selecionados if a in populares]
    
    selected = st.multiselect(
        "Ações Populares",
        options=populares,
        default=default_populares,
        key="multi_populares"
    )
    
    st.session_state.ativos_selecionados = selected
    
    col1, col2 = st.columns(2)
    
    if col1.button("✅ Selecionar Todos", key="btn_todos_pop", use_container_width=True):
        st.session_state.ativos_selecionados = populares
        st.rerun()
    
    if col2.button("🗑️ Limpar", key="btn_limpar_pop", use_container_width=True):
        st.session_state.ativos_selecionados = []
        st.rerun()

with tab3:
    st.info("💡 Use formato: PETR4.SA (Brasil) ou AAPL (EUA)")
    
    manual_input = st.text_area(
        "Digite os tickers (separados por vírgula ou linha)",
        height=100,
        key="manual_input",
        placeholder="PETR4.SA, VALE3.SA, ITUB4.SA\nou\nAAPL\nMSFT\nGOOGL"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_sa = st.checkbox("Adicionar .SA automaticamente", value=False, key="auto_sa")
    
    with col2:
        validar = st.checkbox("Validar ativos", value=True, key="validar")
    
    if manual_input:
        tickers = manual_input.replace(',', ' ').replace('\n', ' ').replace(';', ' ').split()
        tickers = [t.strip().upper() for t in tickers if t.strip()]
        
        if auto_sa:
            tickers = [t if '.' in t else f"{t}.SA" for t in tickers]
        
        tickers = list(set(tickers))
        
        if validar:
            with st.spinner("🔍 Validando ativos..."):
                validos = []
                invalidos = []
                
                progress = st.progress(0)
                
                for i, ticker in enumerate(tickers):
                    try:
                        ativo = yf.Ticker(ticker)
                        hist = ativo.history(period="5d")
                        if not hist.empty:
                            validos.append(ticker)
                        else:
                            invalidos.append(ticker)
                    except:
                        invalidos.append(ticker)
                    
                    progress.progress((i + 1) / len(tickers))
                
                progress.empty()
                
                if validos:
                    st.success(f"✅ Válidos: {', '.join(validos)}")
                    st.session_state.ativos_selecionados = validos
                
                if invalidos:
                    st.error(f"❌ Inválidos: {', '.join(invalidos)}")
        else:
            st.session_state.ativos_selecionados = tickers
            st.success(f"✅ {len(tickers)} ativos adicionados")

# ==================== RESUMO DA SELEÇÃO ====================

st.divider()

ativos_finais = st.session_state.ativos_selecionados

if ativos_finais:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.metric("🎯 Ativos Selecionados", len(ativos_finais))
    
    with col2:
        with st.expander("📋 Ver Lista Completa", expanded=False):
            cols = st.columns(5)
            for i, ativo in enumerate(sorted(ativos_finais)):
                with cols[i % 5]:
                    st.write(f"✓ {ativo}")
    
    with col3:
        if st.button("🗑️ Limpar Tudo", key="btn_limpar_tudo", use_container_width=True):
            st.session_state.ativos_selecionados = []
            st.rerun()
else:
    st.warning("⚠️ Nenhum ativo selecionado. Selecione ativos para continuar.")
    st.stop()

if data_inicio >= data_fim:
    st.error("❌ Data inicial deve ser anterior à data final!")
    st.stop()


# ==================== RESUMO DA SELEÇÃO ====================

st.divider()

ativos_finais = st.session_state.ativos_selecionados

if ativos_finais:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.metric("🎯 Ativos Selecionados", len(ativos_finais))
    
    with col2:
        with st.expander("📋 Ver Lista Completa", expanded=False):
            cols = st.columns(5)
            for i, ativo in enumerate(sorted(ativos_finais)):
                with cols[i % 5]:
                    st.write(f"✓ {ativo}")
    
    with col3:
        if st.button("🗑️ Limpar Tudo", use_container_width=True):
            st.session_state.ativos_selecionados = []
            st.rerun()
else:
    st.warning("⚠️ Nenhum ativo selecionado. Selecione ativos para continuar.")
    st.stop()

if data_inicio >= data_fim:
    st.error("❌ Data inicial deve ser anterior à data final!")
    st.stop()

# ==================== ANÁLISE ====================

st.divider()

if st.button("🚀 INICIAR ANÁLISE COMPLETA", type="primary", use_container_width=True):
    
    # ========== DOWNLOAD DE DADOS ==========
    with st.spinner("📥 Baixando dados dos ativos..."):
        dados_dict, sucessos, erros = baixar_dados_ativos(ativos_finais, data_inicio, data_fim)
        
        if not dados_dict:
            st.error("❌ Não foi possível baixar dados de nenhum ativo!")
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
    
    # ========== DIVIDENDOS ==========
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
    st.header("🎯 Otimização de Carteira")
    
    with st.spinner("🔄 Calculando carteiras ótimas..."):
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
        div_anual_total = 0
        
        if df_dividendos is not None and not df_dividendos.empty:
            pesos_div = calcular_pesos_dividendos(df_dividendos)
            if pesos_div is not None:
                ret_div, vol_div, sharpe_div = calcular_metricas_portfolio(
                    pesos_div, retorno_esperado, matriz_cov, taxa_livre_risco
                )
                div_anual_total = (df_dividendos.sum().sum() / len(df_dividendos)) * 12
    
    # ========== COMPARAÇÃO DAS ESTRATÉGIAS ==========
    st.subheader("📊 Comparação das Estratégias")
    
    estrategias = []
    
    # Sharpe
    estrategias.append({
        'Estratégia': '🏆 Máximo Sharpe Ratio',
        'Objetivo': 'Melhor retorno ajustado ao risco',
        'Retorno (%)': ret_sharpe * 100,
        'Volatilidade (%)': vol_sharpe * 100,
        'Sharpe': sharpe_sharpe,
        'Pesos': pesos_sharpe,
        'Cor': '#FF4B4B'
    })
    
    # Min Vol
    estrategias.append({
        'Estratégia': '🛡️ Mínima Volatilidade',
        'Objetivo': 'Menor risco possível',
        'Retorno (%)': ret_min_vol * 100,
        'Volatilidade (%)': vol_min_vol * 100,
        'Sharpe': sharpe_min_vol,
        'Pesos': pesos_min_vol,
        'Cor': '#00CC00'
    })
    
    # Dividendos
    if pesos_div is not None:
        estrategias.append({
            'Estratégia': '💰 Foco em Dividendos',
            'Objetivo': 'Máxima renda passiva',
            'Retorno (%)': ret_div * 100,
            'Volatilidade (%)': vol_div * 100,
            'Sharpe': sharpe_div,
            'Pesos': pesos_div,
            'Cor': '#FFD700'
        })
    
    # Tabela Comparativa
    df_comp = pd.DataFrame([{
        'Estratégia': e['Estratégia'],
        'Objetivo': e['Objetivo'],
        'Retorno Anual': f"{e['Retorno (%)']:.2f}%",
        'Volatilidade': f"{e['Volatilidade (%)']:.2f}%",
        'Sharpe Ratio': f"{e['Sharpe']:.2f}"
    } for e in estrategias])
    
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    # ========== RECOMENDAÇÃO ==========
    st.subheader("🎖️ Melhor Estratégia")
    
    melhor_sharpe = max(estrategias, key=lambda x: x['Sharpe'])
    menor_risco = min(estrategias, key=lambda x: x['Volatilidade (%)'])
    maior_retorno = max(estrategias, key=lambda x: x['Retorno (%)'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"**🏆 Melhor Sharpe**\n\n{melhor_sharpe['Estratégia']}\n\nSharpe: {melhor_sharpe['Sharpe']:.2f}")
    
    with col2:
        st.info(f"**🛡️ Menor Risco**\n\n{menor_risco['Estratégia']}\n\nVol: {menor_risco['Volatilidade (%)']:.2f}%")
    
    with col3:
        st.warning(f"**📈 Maior Retorno**\n\n{maior_retorno['Estratégia']}\n\nRet: {maior_retorno['Retorno (%)']:.2f}%")
    
    # ========== ANÁLISE POR PERFIL ==========
    st.subheader("👤 Recomendação por Perfil de Investidor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🛡️ CONSERVADOR**
        
        Prioriza segurança e estabilidade.
        
        **Recomendação:**  
        {}
        
        - Menor volatilidade
        - Retorno previsível
        - Menor risco
        """.format(menor_risco['Estratégia']))
    
    with col2:
        st.markdown("""
        **⚖️ MODERADO**
        
        Busca equilíbrio entre risco e retorno.
        
        **Recomendação:**  
        {}
        
        - Melhor relação risco/retorno
        - Diversificação eficiente
        - Sharpe otimizado
        """.format(melhor_sharpe['Estratégia']))
    
    with col3:
        if pesos_div is not None:
            st.markdown("""
            **💰 RENDA PASSIVA**
            
            Foco em geração de renda mensal.
            
            **Recomendação:**  
            💰 Foco em Dividendos
            
            - Dividendos regulares
            - Fluxo de caixa mensal
            - Renda passiva
            """)
        else:
            st.markdown("""
            **🚀 AGRESSIVO**
            
            Busca máximo retorno.
            
            **Recomendação:**  
            {}
            
            - Maior retorno potencial
            - Aceita volatilidade
            - Foco em crescimento
            """.format(maior_retorno['Estratégia']))
    
    # ========== DETALHAMENTO DAS CARTEIRAS ==========
    st.divider()
    st.header("📋 Detalhamento das Carteiras")
    
    tabs = st.tabs([e['Estratégia'] for e in estrategias])
    
    for tab, estrategia in zip(tabs, estrategias):
        with tab:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Tabela de alocação
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
                    }).background_gradient(subset=['Peso (%)'], cmap='Blues'),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                # Pizza
                fig_pie = go.Figure(data=[go.Pie(
                    labels=df_alocacao['Ativo'],
                    values=df_alocacao['Peso (%)'],
                    hole=0.4,
                    marker_colors=[estrategia['Cor']] * len(df_alocacao)
                )])
                fig_pie.update_layout(
                    title=estrategia['Estratégia'],
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📈 Retorno Anual", f"{estrategia['Retorno (%)']:.2f}%")
            with col2:
                st.metric("📊 Volatilidade", f"{estrategia['Volatilidade (%)']:.2f}%")
            with col3:
                st.metric("⚡ Sharpe Ratio", f"{estrategia['Sharpe']:.2f}")
            with col4:
                valor_final = capital_inicial * (1 + estrategia['Retorno (%)'] / 100)
                st.metric("💰 Valor Final (1 ano)", f"R$ {valor_final:,.2f}")
    
    # ========== ANÁLISE DE DIVIDENDOS DETALHADA ==========
    if df_dividendos is not None and not df_dividendos.empty:
        st.divider()
        st.header("💰 Análise Detalhada de Dividendos")
        
        # Gráfico mensal
        fig_div = go.Figure()
        
        for ativo in df_dividendos.columns:
            fig_div.add_trace(go.Bar(
                name=ativo,
                x=df_dividendos.index.strftime('%b/%y'),
                y=df_dividendos[ativo],
                hovertemplate='<b>%{fullData.name}</b><br>R$ %{y:.2f}<extra></extra>'
            ))
        
        fig_div.update_layout(
            title="Dividendos Mensais por Ativo",
            xaxis_title="Mês",
            yaxis_title="Dividendos (R$)",
            barmode='stack',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_div, use_container_width=True)
        
        # Métricas de dividendos
        col1, col2, col3, col4 = st.columns(4)
        
        total_div = df_dividendos.sum().sum()
        media_mensal = df_dividendos.sum(axis=1).mean()
        projecao_anual = media_mensal * 12
        dy_medio = (projecao_anual / capital_inicial) * 100
        
        with col1:
            st.metric("💰 Total Recebido", f"R$ {total_div:,.2f}")
        with col2:
            st.metric("📅 Média Mensal", f"R$ {media_mensal:,.2f}")
        with col3:
            st.metric("📈 Projeção Anual", f"R$ {projecao_anual:,.2f}")
        with col4:
            st.metric("📊 DY Médio", f"{dy_medio:.2f}%")
    
    # ========== FRONTEIRA EFICIENTE ==========
    st.divider()
    st.header("📈 Fronteira Eficiente de Markowitz")
    
    with st.spinner("Calculando fronteira eficiente..."):
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
    
    # Pontos simulados
    fig_front.add_trace(go.Scatter(
        x=resultados[1, :] * 100,
        y=resultados[0, :] * 100,
        mode='markers',
        marker=dict(
            size=4,
            color=resultados[2, :],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe")
        ),
        name='Carteiras Possíveis',
        hovertemplate='Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>'
    ))
    
    # Carteiras ótimas
    for estrategia in estrategias:
        fig_front.add_trace(go.Scatter(
            x=[estrategia['Volatilidade (%)']],
            y=[estrategia['Retorno (%)']],
            mode='markers',
            marker=dict(size=20, color=estrategia['Cor'], symbol='star', line=dict(width=2, color='white')),
            name=estrategia['Estratégia'],
            hovertemplate=f"<b>{estrategia['Estratégia']}</b><br>" +
                         f"Retorno: {estrategia['Retorno (%)']:.2f}%<br>" +
                         f"Volatilidade: {estrategia['Volatilidade (%)']:.2f}%<br>" +
                         f"Sharpe: {estrategia['Sharpe']:.2f}<extra></extra>"
        ))
    
    fig_front.update_layout(
        title="Fronteira Eficiente - Todas as Estratégias",
        xaxis_title="Volatilidade Anual (%)",
        yaxis_title="Retorno Anual (%)",
        height=600,
        hovermode='closest',
        showlegend=True
    )
    
    st.plotly_chart(fig_front, use_container_width=True)
    
    # ========== SIMULAÇÃO HISTÓRICA ==========
    st.divider()
    st.header("📊 Simulação de Performance Histórica")
    
    fig_perf = go.Figure()
    
    for estrategia in estrategias:
        retornos_estrategia = (retornos * estrategia['Pesos']).sum(axis=1)
        valor_estrategia = capital_inicial * (1 + retornos_estrategia).cumprod()
        
        fig_perf.add_trace(go.Scatter(
            x=valor_estrategia.index,
            y=valor_estrategia.values,
            mode='lines',
            name=estrategia['Estratégia'],
            line=dict(width=3, color=estrategia['Cor']),
            hovertemplate='<b>%{fullData.name}</b><br>R$ %{y:,.2f}<extra></extra>'
        ))
    
    fig_perf.update_layout(
        title="Evolução do Valor da Carteira",
        xaxis_title="Data",
        yaxis_title="Valor (R$)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # ========== DOWNLOAD ==========
    st.divider()
    st.header("💾 Exportar Resultados")
    
    cols = st.columns(len(estrategias))
    
    for col, estrategia in zip(cols, estrategias):
        with col:
            df_export = pd.DataFrame({
                'Ativo': ativos_com_dados,
                'Peso (%)': estrategia['Pesos'] * 100,
                'Valor (R$)': estrategia['Pesos'] * capital_inicial
            })
            df_export = df_export[df_export['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=False)
            
            csv = df_export.to_csv(index=False).encode('utf-8')
            nome_arquivo = estrategia['Estratégia'].replace('🏆 ', '').replace('🛡️ ', '').replace('💰 ', '')
            nome_arquivo = nome_arquivo.replace(' ', '_').lower()
            
            st.download_button(
                label=f"📥 {estrategia['Estratégia']}",
                data=csv,
                file_name=f"{nome_arquivo}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==================== RODAPÉ ====================
st.divider()
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <p style='font-size: 0.9rem;'>
        📊 <b>Otimizador de Carteira - Teoria Moderna de Portfólio (Markowitz)</b><br>
        Dados fornecidos por Yahoo Finance | Apenas para fins educacionais<br>
        ⚠️ Não constitui recomendação de investimento
    </p>
</div>
""", unsafe_allow_html=True)
