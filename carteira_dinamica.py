import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ativos_b3

# Configuração da página
st.set_page_config(
    page_title="Otimização de Carteira Dinâmica",
    page_icon="📈",
    layout="wide"
)

# Título
st.title("📈 Otimização de Carteira de Investimentos")
st.markdown("---")

# Sidebar com informações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Período de análise
    st.subheader("Período de Análise")
    data_inicio = st.date_input(
        "Data Inicial",
        value=datetime.now() - timedelta(days=365),
        max_value=datetime.now()
    )
    data_fim = st.date_input(
        "Data Final",
        value=datetime.now(),
        max_value=datetime.now()
    )
    
    # Capital inicial
    st.subheader("Capital")
    capital_inicial = st.number_input(
        "Capital Inicial (R$)",
        min_value=100.0,
        value=10000.0,
        step=100.0,
        format="%.2f"
    )
    
    # Taxa livre de risco
    taxa_livre_risco = st.number_input(
        "Taxa Livre de Risco Anual (%)",
        min_value=0.0,
        value=13.75,
        step=0.25,
        format="%.2f"
    ) / 100
    
    # Opções de análise
    st.subheader("Análises")
    incluir_dividendos = st.checkbox("Incluir Análise de Dividendos", value=True)

# Seção de seleção de ativos
st.subheader("📊 Seleção de Ativos")

# Escolher método de seleção
metodo_selecao = st.radio(
    "Como você deseja selecionar os ativos?",
    ["🔍 Buscar por Tipo e Segmento", "📋 Lista Rápida de Ações Populares", "✏️ Adicionar Manualmente"],
    horizontal=True
)

ativos_finais = []

# ============= MÉTODO 1: BUSCA POR SEGMENTO =============
if metodo_selecao == "🔍 Buscar por Tipo e Segmento":
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Seletor de tipo de ativo
        tipo_ativo = st.selectbox(
            "**1. Escolha o Tipo de Ativo:**",
            ["Ações", "FIIs (Fundos Imobiliários)", "ETFs", "BDRs"]
        )
        
        # Mapear seleção para os dados
        tipo_map = {
            "Ações": ativos_b3.ACOES_B3,
            "FIIs (Fundos Imobiliários)": ativos_b3.FIIS,
            "ETFs": ativos_b3.ETFS,
            "BDRs": ativos_b3.BDRS
        }
        
        dados_tipo = tipo_map[tipo_ativo]
        
        # Seletor de segmentos
        st.write("**2. Escolha os Segmentos:**")
        segmentos_selecionados = st.multiselect(
            "Segmentos disponíveis:",
            options=list(dados_tipo.keys()),
            help="Selecione um ou mais segmentos para ver os ativos disponíveis"
        )
    
    with col2:
        if segmentos_selecionados:
            st.write("**3. Selecione os Ativos Específicos:**")
            
            # Coletar ativos dos segmentos selecionados
            ativos_disponiveis = []
            for segmento in segmentos_selecionados:
                ativos_disponiveis.extend(dados_tipo[segmento])
            
            # Remover duplicatas e ordenar
            ativos_disponiveis = sorted(list(set(ativos_disponiveis)))
            
            # Mostrar informação
            st.info(f"📊 {len(ativos_disponiveis)} ativos disponíveis nos segmentos selecionados")
            
            # Multiselect para escolher ativos específicos
            ativos_finais = st.multiselect(
                "Escolha os ativos que deseja analisar:",
                options=ativos_disponiveis,
                default=ativos_disponiveis[:min(5, len(ativos_disponiveis))],
                help="Selecione os ativos específicos para análise"
            )
            
            # Botões de ação
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if st.button("✅ Selecionar Todos", use_container_width=True):
                    ativos_finais = ativos_disponiveis
                    st.rerun()
            with col_btn2:
                if st.button("🔄 Limpar Seleção", use_container_width=True):
                    ativos_finais = []
                    st.rerun()
            with col_btn3:
                if st.button("🎲 Aleatórios (10)", use_container_width=True):
                    import random
                    ativos_finais = random.sample(ativos_disponiveis, min(10, len(ativos_disponiveis)))
                    st.rerun()
        else:
            st.info("👈 Selecione um ou mais segmentos na coluna à esquerda para ver os ativos disponíveis")

# ============= MÉTODO 2: LISTA RÁPIDA =============
elif metodo_selecao == "📋 Lista Rápida de Ações Populares":
    st.markdown("---")
    st.write("Selecione rapidamente entre as ações mais negociadas da B3:")
    
    ativos_populares = [
        'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
        'WEGE3.SA', 'RENT3.SA', 'MGLU3.SA', 'B3SA3.SA', 'SUZB3.SA',
        'BBAS3.SA', 'SANB11.SA', 'EGIE3.SA', 'CPLE6.SA', 'VIVT3.SA',
        'RADL3.SA', 'RAIL3.SA', 'EMBR3.SA', 'CSNA3.SA', 'USIM5.SA',
        'JBSS3.SA', 'LREN3.SA', 'HAPV3.SA', 'PRIO3.SA', 'KLBN11.SA'
    ]
    
    ativos_finais = st.multiselect(
        "Escolha as ações:",
        options=sorted(ativos_populares),
        default=['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'WEGE3.SA'],
        help="Selecione os ativos que deseja incluir na análise"
    )
    
    # Botões de ação
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Selecionar Todos", use_container_width=True):
            ativos_finais = ativos_populares
            st.rerun()
    with col2:
        if st.button("🔄 Limpar Seleção", use_container_width=True):
            ativos_finais = []
            st.rerun()

# ============= MÉTODO 3: ADICIONAR MANUALMENTE =============
else:  # Adicionar Manualmente
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("💡 **Dicas:**\n- Para ações brasileiras, adicione '.SA' (ex: PETR4.SA)\n- Para ações americanas, use apenas o ticker (ex: AAPL, MSFT)\n- Separe múltiplos ativos por vírgula, espaço ou quebra de linha")
        
        ativos_manuais_input = st.text_area(
            "Digite os tickers dos ativos:",
            placeholder="Exemplos:\n\nPETR4.SA, VALE3.SA, ITUB4.SA\n\nou\n\nAAPL\nMSFT\nGOOGL\nTSLA",
            height=150
        )
    
    with col2:
        adicionar_sufixo = st.checkbox(
            "Adicionar '.SA' automaticamente",
            value=False,
            help="Adiciona '.SA' ao final de todos os tickers"
        )
        
        validar_ativos = st.checkbox(
            "Validar ativos",
            value=True,
            help="Verifica se os ativos existem antes de usar"
        )
    
    # Processar ativos manuais
    if ativos_manuais_input:
        # Limpar e processar input
        ativos_temp = ativos_manuais_input.replace(',', ' ').replace('\n', ' ').replace(';', ' ').split()
        ativos_manuais = [ativo.strip().upper() for ativo in ativos_temp if ativo.strip()]
        
        # Adicionar sufixo se solicitado
        if adicionar_sufixo:
            ativos_manuais = [
                ativo if '.' in ativo else f"{ativo}.SA" 
                for ativo in ativos_manuais
            ]
        
        # Remover duplicatas
        ativos_manuais = list(set(ativos_manuais))
        
        if ativos_manuais:
            st.success(f"✅ {len(ativos_manuais)} ativo(s) detectado(s): {', '.join(ativos_manuais)}")
            
            # Validar ativos se solicitado
            if validar_ativos:
                with st.spinner("🔍 Validando ativos..."):
                    ativos_validos = []
                    ativos_invalidos = []
                    
                    progress_bar = st.progress(0)
                    
                    for i, ativo in enumerate(ativos_manuais):
                        try:
                            ticker = yf.Ticker(ativo)
                            hist = ticker.history(period="5d")
                            if not hist.empty and len(hist) > 0:
                                ativos_validos.append(ativo)
                            else:
                                ativos_invalidos.append(ativo)
                        except:
                            ativos_invalidos.append(ativo)
                        
                        progress_bar.progress((i + 1) / len(ativos_manuais))
                    
                    progress_bar.empty()
                    
                    if ativos_validos:
                        st.success(f"✅ **Ativos válidos ({len(ativos_validos)}):** {', '.join(ativos_validos)}")
                        ativos_finais = ativos_validos
                    
                    if ativos_invalidos:
                        st.error(f"❌ **Ativos inválidos ({len(ativos_invalidos)}):** {', '.join(ativos_invalidos)}")
            else:
                ativos_finais = ativos_manuais

# ============= RESUMO DA SELEÇÃO =============
st.markdown("---")

if ativos_finais:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("🎯 Total de Ativos Selecionados", len(ativos_finais))
    
    with col2:
        with st.expander("📋 Ver lista completa de ativos", expanded=True):
            # Mostrar em colunas
            num_cols = 5
            cols = st.columns(num_cols)
            for i, ativo in enumerate(sorted(ativos_finais)):
                with cols[i % num_cols]:
                    st.write(f"✓ {ativo}")
else:
    st.warning("⚠️ **Nenhum ativo selecionado!** Por favor, selecione ativos para continuar com a análise.")
    st.stop()

# Verificar se o período é válido
if data_inicio >= data_fim:
    st.error("❌ A data inicial deve ser anterior à data final!")
    st.stop()

# ============= BOTÃO PARA INICIAR ANÁLISE =============
st.markdown("---")
if st.button("🚀 Iniciar Análise e Otimização", type="primary", use_container_width=True):
    
    # Baixar dados
    st.subheader("📥 Baixando Dados dos Ativos")
    
    with st.spinner("Baixando dados históricos..."):
        try:
            if len(ativos_finais) == 1:
                ticker = yf.Ticker(ativos_finais[0])
                dados_temp = ticker.history(start=data_inicio, end=data_fim)
                if dados_temp.empty:
                    st.error(f"❌ Não foi possível obter dados para {ativos_finais[0]}")
                    st.stop()
                dados = pd.DataFrame({ativos_finais[0]: dados_temp['Close']})
            else:
                dados = yf.download(
                    ativos_finais,
                    start=data_inicio,
                    end=data_fim,
                    progress=False
                )
                
                if dados.empty:
                    st.error("❌ Não foi possível obter dados para os ativos selecionados.")
                    st.stop()
                
                if isinstance(dados.columns, pd.MultiIndex):
                    if 'Adj Close' in dados.columns.get_level_values(0):
                        dados = dados['Adj Close']
                    elif 'Close' in dados.columns.get_level_values(0):
                        dados = dados['Close']
                    else:
                        dados = dados.iloc[:, dados.columns.get_level_values(0) == dados.columns.get_level_values(0)[0]]
                        dados.columns = dados.columns.droplevel(0)
            
            if not isinstance(dados, pd.DataFrame):
                dados = dados.to_frame()
                if len(ativos_finais) == 1:
                    dados.columns = ativos_finais
            
            dados = dados.dropna(axis=1, how='all')
            threshold = len(dados.columns) * 0.5
            dados = dados.dropna(thresh=threshold)
            dados = dados.ffill().bfill()
            
            if dados.empty:
                st.error("❌ Não foi possível obter dados válidos para os ativos selecionados no período especificado.")
                st.stop()
            
            ativos_com_dados = dados.columns.tolist()
            
            if len(ativos_com_dados) < len(ativos_finais):
                ativos_sem_dados = set(ativos_finais) - set(ativos_com_dados)
                st.warning(f"⚠️ Ativos sem dados removidos: {', '.join(ativos_sem_dados)}")
            
            if len(ativos_com_dados) < 2:
                st.error("❌ É necessário pelo menos 2 ativos com dados válidos.")
                st.stop()
            
            st.success(f"✅ Dados baixados para {len(ativos_com_dados)} ativos!")
            
            with st.expander("👁️ Visualizar dados históricos"):
                st.write(f"**Período:** {dados.index[0].strftime('%d/%m/%Y')} até {dados.index[-1].strftime('%d/%m/%Y')}")
                st.write(f"**Total de dias:** {len(dados)}")
                st.dataframe(dados.tail(10).style.format('{:.2f}'), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Erro ao baixar dados: {str(e)}")
            st.stop()
    
    # Análise de Dividendos
    if incluir_dividendos:
        st.markdown("---")
        st.subheader("💰 Análise de Dividendos")
        
        with st.spinner("Coletando dados de dividendos..."):
            try:
                dividendos_data = {}
                
                for ativo in ativos_com_dados:
                    ticker = yf.Ticker(ativo)
                    divs = ticker.dividends
                    
                    if not divs.empty:
                        divs = divs[(divs.index >= pd.Timestamp(data_inicio)) & (divs.index <= pd.Timestamp(data_fim))]
                        if not divs.empty:
                            dividendos_data[ativo] = divs
                
                if dividendos_data:
                    # Criar DataFrame com dividendos mensais
                    df_dividendos = pd.DataFrame()
                    
                    for ativo, divs in dividendos_data.items():
                        divs_mensais = divs.resample('M').sum()
                        df_dividendos[ativo] = divs_mensais
                    
                    df_dividendos = df_dividendos.fillna(0)
                    
                    # ========== GRÁFICO DE BARRAS MENSAIS ==========
                    st.write("### 📊 Distribuição Mensal de Dividendos por Ativo")
                    
                    fig_div = go.Figure()
                    
                    for ativo in df_dividendos.columns:
                        fig_div.add_trace(go.Bar(
                            name=ativo,
                            x=df_dividendos.index.strftime('%b/%Y'),
                            y=df_dividendos[ativo],
                            text=df_dividendos[ativo].apply(lambda x: f'R$ {x:.2f}' if x > 0 else ''),
                            textposition='inside',
                            hovertemplate='<b>%{fullData.name}</b><br>R$ %{y:.2f}<extra></extra>'
                        ))
                    
                    fig_div.update_layout(
                        title="Dividendos Mensais Recebidos por Ativo",
                        xaxis_title="Mês",
                        yaxis_title="Dividendos (R$)",
                        barmode='stack',
                        height=500,
                        hovermode='x unified',
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
                    st.write("### 📅 Tabela de Dividendos Mensais (R$)")
                    
                    # Formatar tabela
                    df_div_display = df_dividendos.copy()
                    df_div_display.index = df_div_display.index.strftime('%b/%Y')
                    
                    # Adicionar linha de total
                    df_div_display.loc['TOTAL'] = df_div_display.sum()
                    
                    # Adicionar coluna de total por mês
                    df_div_display['Total Mensal'] = df_div_display.sum(axis=1)
                    
                    st.dataframe(
                        df_div_display.style.format('R$ {:.2f}').background_gradient(cmap='Greens', axis=None),
                        use_container_width=True
                    )
                    
                    # ========== RESUMO ESTATÍSTICO ==========
                    st.write("### 📈 Resumo Estatístico de Dividendos")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**💰 Total de Dividendos por Ativo**")
                        div_totais = df_dividendos.sum().sort_values(ascending=False)
                        df_totais = pd.DataFrame({
                            'Ativo': div_totais.index,
                            'Total (R$)': div_totais.values,
                            '% do Total': (div_totais.values / div_totais.sum() * 100)
                        })
                        st.dataframe(
                            df_totais.style.format({
                                'Total (R$)': 'R$ {:.2f}',
                                '% do Total': '{:.1f}%'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col2:
                        st.write("**📊 Dividendos Médios Mensais**")
                        div_medios = df_dividendos.mean().sort_values(ascending=False)
                        df_medios = pd.DataFrame({
                            'Ativo': div_medios.index,
                            'Média Mensal (R$)': div_medios.values,
                            'Anualizado (R$)': div_medios.values * 12
                        })
                        st.dataframe(
                            df_medios.style.format({
                                'Média Mensal (R$)': 'R$ {:.2f}',
                                'Anualizado (R$)': 'R$ {:.2f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col3:
                        st.write("**📈 Dividend Yield**")
                        dy_data = []
                        for ativo in df_dividendos.columns:
                            preco_medio = dados[ativo].mean()
                            div_total = df_dividendos[ativo].sum()
                            div_anual = (div_total / len(df_dividendos)) * 12
                            dy_mensal = (div_total / len(df_dividendos) / preco_medio) * 100 if preco_medio > 0 else 0
                            dy_anual = (div_anual / preco_medio) * 100 if preco_medio > 0 else 0
                            dy_data.append({
                                'Ativo': ativo,
                                'DY Mensal (%)': dy_mensal,
                                'DY Anual (%)': dy_anual
                            })
                        
                        df_dy = pd.DataFrame(dy_data).sort_values('DY Anual (%)', ascending=False)
                        st.dataframe(
                            df_dy.style.format({
                                'DY Mensal (%)': '{:.2f}%',
                                'DY Anual (%)': '{:.2f}%'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # ========== MÉTRICAS GERAIS ==========
                    st.write("### 🎯 Métricas Gerais de Dividendos")
                    
                    total_dividendos = df_dividendos.sum().sum()
                    media_mensal = df_dividendos.sum(axis=1).mean()
                    projecao_anual = media_mensal * 12
                    meses_com_dividendos = (df_dividendos.sum(axis=1) > 0).sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "💰 Total Recebido",
                            f"R$ {total_dividendos:,.2f}",
                            help="Total de dividendos recebidos no período"
                        )
                    
                    with col2:
                        st.metric(
                            "📅 Média Mensal",
                            f"R$ {media_mensal:,.2f}",
                            help="Média de dividendos por mês"
                        )
                    
                    with col3:
                        st.metric(
                            "📈 Projeção Anual",
                            f"R$ {projecao_anual:,.2f}",
                            help="Projeção anual baseada na média mensal"
                        )
                    
                    with col4:
                        st.metric(
                            "✅ Meses Pagantes",
                            f"{meses_com_dividendos} de {len(df_dividendos)}",
                            help="Meses que receberam dividendos"
                        )
                    
                    # ========== GRÁFICO DE DIVIDENDOS ACUMULADOS ==========
                    st.write("### 📈 Evolução dos Dividendos Acumulados")
                    
                    df_div_acum = df_dividendos.cumsum()
                    
                    fig_div_acum = go.Figure()
                    
                    for ativo in df_div_acum.columns:
                        fig_div_acum.add_trace(go.Scatter(
                            name=ativo,
                            x=df_div_acum.index.strftime('%b/%Y'),
                            y=df_div_acum[ativo],
                            mode='lines+markers',
                            line=dict(width=2),
                            marker=dict(size=6),
                            hovertemplate='<b>%{fullData.name}</b><br>Acumulado: R$ %{y:.2f}<extra></extra>'
                        ))
                    
                    fig_div_acum.update_layout(
                        title="Dividendos Acumulados ao Longo do Tempo",
                        xaxis_title="Mês",
                        yaxis_title="Dividendos Acumulados (R$)",
                        height=450,
                        hovermode='x unified',
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    
                    st.plotly_chart(fig_div_acum, use_container_width=True)
                    
                    # ========== COMPARAÇÃO: DIVIDENDOS vs CARTEIRAS OTIMIZADAS ==========
                    st.write("### ⚖️ Análise Comparativa: Estratégias de Investimento")
                    
                    st.info("""
                    **📊 Comparação de Estratégias:**
                    
                    Vamos comparar três abordagens diferentes:
                    1. **Foco em Dividendos**: Manter os ativos atuais priorizando renda passiva
                    2. **Máximo Sharpe Ratio**: Otimizar para melhor relação retorno/risco
                    3. **Mínima Volatilidade**: Otimizar para menor risco
                    
                    Esta análise será calculada após a otimização das carteiras abaixo.
                    """)
                    
                    # Calcular carteira focada em dividendos (pesos proporcionais aos dividendos)
                    div_totais_norm = df_dividendos.sum()
                    if div_totais_norm.sum() > 0:
                        pesos_dividendos = div_totais_norm / div_totais_norm.sum()
                        
                        # Criar DataFrame para comparação posterior
                        df_estrategia_div = pd.DataFrame({
                            'Ativo': pesos_dividendos.index,
                            'Peso (%)': pesos_dividendos.values * 100,
                            'Dividendos Anuais Estimados (R$)': (df_dividendos.sum() / len(df_dividendos) * 12).values
                        })
                        
                        st.write("**💰 Carteira Focada em Dividendos (Pesos Proporcionais aos Dividendos)**")
                        st.dataframe(
                            df_estrategia_div.style.format({
                                'Peso (%)': '{:.2f}%',
                                'Dividendos Anuais Estimados (R$)': 'R$ {:.2f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Calcular métricas da carteira de dividendos
                        retornos_div = (retornos * pesos_dividendos.values).sum(axis=1)
                        ret_div = retornos_div.mean() * 252
                        vol_div = retornos_div.std() * np.sqrt(252)
                        sharpe_div = (ret_div - taxa_livre_risco) / vol_div
                        
                        dividendos_anuais_estimados = (df_dividendos.sum().sum() / len(df_dividendos)) * 12
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Retorno Anual", f"{ret_div*100:.2f}%")
                        with col2:
                            st.metric("Volatilidade", f"{vol_div*100:.2f}%")
                        with col3:
                            st.metric("Sharpe Ratio", f"{sharpe_div:.2f}")
                        with col4:
                            st.metric("Dividendos/Ano", f"R$ {dividendos_anuais_estimados:,.2f}")
                        
                        # Salvar para comparação posterior
                        st.session_state['carteira_dividendos'] = {
                            'pesos': pesos_dividendos.values,
                            'retorno': ret_div,
                            'volatilidade': vol_div,
                            'sharpe': sharpe_div,
                            'dividendos_anuais': dividendos_anuais_estimados,
                            'df': df_estrategia_div
                        }
                    
                else:
                    st.info("ℹ️ Nenhum dividendo encontrado para os ativos selecionados no período analisado.")
                    st.warning("""
                    **Possíveis razões:**
                    - Os ativos não pagaram dividendos no período
                    - Dados de dividendos não disponíveis no Yahoo Finance
                    - Período de análise muito curto
                    """)
                    
            except Exception as e:
                st.warning(f"⚠️ Não foi possível coletar dados de dividendos: {str(e)}")
                st.info("A análise continuará sem os dados de dividendos.")

    
    # Análise de Retornos
    st.markdown("---")
    st.subheader("📊 Análise de Retornos")
    
    retornos = dados.pct_change().dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Retorno Médio Diário (%)**")
        retorno_medio = (retornos.mean() * 100).sort_values(ascending=False)
        st.dataframe(retorno_medio.to_frame('Retorno (%)').style.format('{:.4f}'), use_container_width=True)
    
    with col2:
        st.write("**Volatilidade (%)**")
        volatilidade = (retornos.std() * 100).sort_values(ascending=False)
        st.dataframe(volatilidade.to_frame('Volatilidade (%)').style.format('{:.4f}'), use_container_width=True)
    
    # Matriz de Correlação
    st.write("**Matriz de Correlação**")
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
        title="Matriz de Correlação entre Ativos",
        height=600
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Otimização
    st.markdown("---")
    st.subheader("🎯 Otimização da Carteira")
    
    retorno_esperado = retornos.mean()
    matriz_cov = retornos.cov()
    num_ativos = len(ativos_com_dados)
    
    def volatilidade_portfolio(pesos, matriz_cov):
        return np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos))) * np.sqrt(252)
    
    def retorno_portfolio(pesos, retorno_esperado):
        return np.sum(retorno_esperado * pesos) * 252
    
    def sharpe_ratio_negativo(pesos, retorno_esperado, matriz_cov, taxa_livre_risco):
        ret = retorno_portfolio(pesos, retorno_esperado)
        vol = volatilidade_portfolio(pesos, matriz_cov)
        return -(ret - taxa_livre_risco) / vol
    
    restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limites = tuple((0, 1) for _ in range(num_ativos))
    pesos_iniciais = np.array([1/num_ativos] * num_ativos)
    
    with st.spinner("Otimizando..."):
        resultado_sharpe = minimize(
            sharpe_ratio_negativo,
            pesos_iniciais,
            args=(retorno_esperado, matriz_cov, taxa_livre_risco),
            method='SLSQP',
            bounds=limites,
            constraints=restricoes
        )
        
        resultado_min_vol = minimize(
            volatilidade_portfolio,
            pesos_iniciais,
            args=(matriz_cov,),
            method='SLSQP',
            bounds=limites,
            constraints=restricoes
        )
    
    pesos_sharpe = resultado_sharpe.x
    pesos_min_vol = resultado_min_vol.x
    
    ret_sharpe = retorno_portfolio(pesos_sharpe, retorno_esperado)
    vol_sharpe = volatilidade_portfolio(pesos_sharpe, matriz_cov)
    sharpe_sharpe = (ret_sharpe - taxa_livre_risco) / vol_sharpe
    
    ret_min_vol = retorno_portfolio(pesos_min_vol, retorno_esperado)
    vol_min_vol = volatilidade_portfolio(pesos_min_vol, matriz_cov)
    sharpe_min_vol = (ret_min_vol - taxa_livre_risco) / vol_min_vol
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🏆 Máximo Sharpe Ratio")
        st.metric("Retorno Anual", f"{ret_sharpe*100:.2f}%")
        st.metric("Volatilidade", f"{vol_sharpe*100:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe_sharpe:.2f}")
        
        df_sharpe = pd.DataFrame({
            'Ativo': ativos_com_dados,
            'Peso (%)': pesos_sharpe * 100,
            'Valor (R$)': pesos_sharpe * capital_inicial
        })
        df_sharpe = df_sharpe[df_sharpe['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=False)
        st.dataframe(df_sharpe.style.format({
            'Peso (%)': '{:.2f}',
            'Valor (R$)': '{:.2f}'
        }), use_container_width=True)
        
        fig_pie_sharpe = go.Figure(data=[go.Pie(
            labels=df_sharpe['Ativo'],
            values=df_sharpe['Peso (%)'],
            hole=0.3
        )])
        fig_pie_sharpe.update_layout(title="Distribuição")
        st.plotly_chart(fig_pie_sharpe, use_container_width=True)
    
    with col2:
        st.write("### 🛡️ Mínima Volatilidade")
        st.metric("Retorno Anual", f"{ret_min_vol*100:.2f}%")
        st.metric("Volatilidade", f"{vol_min_vol*100:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe_min_vol:.2f}")
        
        df_min_vol = pd.DataFrame({
            'Ativo': ativos_com_dados,
            'Peso (%)': pesos_min_vol * 100,
            'Valor (R$)': pesos_min_vol * capital_inicial
        })
        df_min_vol = df_min_vol[df_min_vol['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=False)
        st.dataframe(df_min_vol.style.format({
            'Peso (%)': '{:.2f}',
            'Valor (R$)': '{:.2f}'
        }), use_container_width=True)
        
        fig_pie_min = go.Figure(data=[go.Pie(
            labels=df_min_vol['Ativo'],
            values=df_min_vol['Peso (%)'],
            hole=0.3
        )])
        fig_pie_min.update_layout(title="Distribuição")
        st.plotly_chart(fig_pie_min, use_container_width=True)

    
        # ========== COMPARAÇÃO FINAL DAS TRÊS ESTRATÉGIAS ==========
    if incluir_dividendos and 'carteira_dividendos' in st.session_state:
        st.markdown("---")
        st.subheader("🎯 Comparação Final: Qual Estratégia é Melhor?")
        
        cart_div = st.session_state['carteira_dividendos']
        
        # Criar tabela comparativa
        df_comparacao_estrategias = pd.DataFrame({
            'Estratégia': [
                '💰 Foco em Dividendos',
                '🏆 Máximo Sharpe Ratio',
                '🛡️ Mínima Volatilidade'
            ],
            'Retorno Anual (%)': [
                cart_div['retorno'] * 100,
                ret_sharpe * 100,
                ret_min_vol * 100
            ],
            'Volatilidade (%)': [
                cart_div['volatilidade'] * 100,
                vol_sharpe * 100,
                vol_min_vol * 100
            ],
            'Sharpe Ratio': [
                cart_div['sharpe'],
                sharpe_sharpe,
                sharpe_min_vol
            ],
            'Dividendos Anuais (R$)': [
                cart_div['dividendos_anuais'],
                0,  # Será calculado
                0   # Será calculado
            ]
        })
        
        # Calcular dividendos estimados para as carteiras otimizadas
        if dividendos_data:
            df_div_anual = (df_dividendos.sum() / len(df_dividendos)) * 12
            
            div_sharpe = (pesos_sharpe * df_div_anual.values).sum()
            div_min_vol = (pesos_min_vol * df_div_anual.values).sum()
            
            df_comparacao_estrategias.loc[1, 'Dividendos Anuais (R$)'] = div_sharpe
            df_comparacao_estrategias.loc[2, 'Dividendos Anuais (R$)'] = div_min_vol
        
        # Adicionar rendimento total (retorno + dividendos)
        df_comparacao_estrategias['Rendimento Total (%)'] = (
            df_comparacao_estrategias['Retorno Anual (%)'] + 
            (df_comparacao_estrategias['Dividendos Anuais (R$)'] / capital_inicial * 100)
        )
        
        st.write("### 📊 Tabela Comparativa das Estratégias")
        st.dataframe(
            df_comparacao_estrategias.style.format({
                'Retorno Anual (%)': '{:.2f}%',
                'Volatilidade (%)': '{:.2f}%',
                'Sharpe Ratio': '{:.2f}',
                'Dividendos Anuais (R$)': 'R$ {:.2f}',
                'Rendimento Total (%)': '{:.2f}%'
            }).background_gradient(subset=['Sharpe Ratio', 'Rendimento Total (%)'], cmap='RdYlGn'),
            use_container_width=True,
            hide_index=True
        )
        
        # Gráfico de radar comparando as estratégias
        st.write("### 🎯 Visualização Comparativa")
        
        fig_radar = go.Figure()
        
        categories = ['Retorno', 'Sharpe Ratio', 'Dividendos', 'Estabilidade']
        
        # Normalizar valores para 0-100
        max_ret = df_comparacao_estrategias['Retorno Anual (%)'].max()
        max_sharpe = df_comparacao_estrategias['Sharpe Ratio'].max()
        max_div = df_comparacao_estrategias['Dividendos Anuais (R$)'].max()
        min_vol = df_comparacao_estrategias['Volatilidade (%)'].min()
        
        for idx, row in df_comparacao_estrategias.iterrows():
            values = [
                (row['Retorno Anual (%)'] / max_ret * 100) if max_ret > 0 else 0,
                (row['Sharpe Ratio'] / max_sharpe * 100) if max_sharpe > 0 else 0,
                (row['Dividendos Anuais (R$)'] / max_div * 100) if max_div > 0 else 0,
                (min_vol / row['Volatilidade (%)'] * 100) if row['Volatilidade (%)'] > 0 else 0
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Fechar o polígono
                theta=categories + [categories[0]],
                fill='toself',
                name=row['Estratégia']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Comparação Normalizada das Estratégias (0-100)",
            height=500
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Análise e recomendação
        st.write("### 💡 Análise e Recomendação")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 Melhor para cada objetivo:**")
            
            melhor_retorno = df_comparacao_estrategias.loc[df_comparacao_estrategias['Retorno Anual (%)'].idxmax(), 'Estratégia']
            melhor_sharpe = df_comparacao_estrategias.loc[df_comparacao_estrategias['Sharpe Ratio'].idxmax(), 'Estratégia']
            melhor_dividendos = df_comparacao_estrategias.loc[df_comparacao_estrategias['Dividendos Anuais (R$)'].idxmax(), 'Estratégia']
            menor_risco = df_comparacao_estrategias.loc[df_comparacao_estrategias['Volatilidade (%)'].idxmin(), 'Estratégia']
            melhor_total = df_comparacao_estrategias.loc[df_comparacao_estrategias['Rendimento Total (%)'].idxmax(), 'Estratégia']
            
            st.success(f"**📈 Maior Retorno:** {melhor_retorno}")
            st.success(f"**⚡ Melhor Sharpe:** {melhor_sharpe}")
            st.success(f"**💰 Mais Dividendos:** {melhor_dividendos}")
            st.success(f"**🛡️ Menor Risco:** {menor_risco}")
            st.success(f"**🏆 Melhor Rendimento Total:** {melhor_total}")
        
        with col2:
            st.write("**📝 Perfis de Investidor:**")
            
            st.info("""
            **Conservador:** 🛡️ Mínima Volatilidade
            - Prioriza segurança e estabilidade
            - Aceita retornos menores
            
            **Moderado:** 💰 Foco em Dividendos
            - Busca renda passiva regular
            - Equilíbrio entre risco e retorno
            
            **Agressivo:** 🏆 Máximo Sharpe
            - Busca máxima eficiência
            - Aceita mais volatilidade
            """)
        
        # Simulação mensal comparativa
        st.write("### 📅 Simulação de Renda Mensal")
        
        renda_mensal_div = cart_div['dividendos_anuais'] / 12
        renda_mensal_sharpe = div_sharpe / 12 if 'div_sharpe' in locals() else 0
        renda_mensal_min_vol = div_min_vol / 12 if 'div_min_vol' in locals() else 0
        
        df_renda_mensal = pd.DataFrame({
            'Estratégia': [
                '💰 Foco em Dividendos',
                '🏆 Máximo Sharpe Ratio',
                '🛡️ Mínima Volatilidade'
            ],
            'Renda Mensal (R$)': [
                renda_mensal_div,
                renda_mensal_sharpe,
                renda_mensal_min_vol
            ],
            'Renda Anual (R$)': [
                cart_div['dividendos_anuais'],
                div_sharpe if 'div_sharpe' in locals() else 0,
                div_min_vol if 'div_min_vol' in locals() else 0
            ]
        })
        
        st.dataframe(
            df_renda_mensal.style.format({
                'Renda Mensal (R$)': 'R$ {:.2f}',
                'Renda Anual (R$)': 'R$ {:.2f}'
            }).background_gradient(subset=['Renda Mensal (R$)'], cmap='Greens'),
            use_container_width=True,
            hide_index=True
        )

    # Fronteira Eficiente
    st.markdown("---")
    st.subheader("📈 Fronteira Eficiente")
    
    with st.spinner("Calculando..."):
        num_portfolios = 5000
        resultados = np.zeros((3, num_portfolios))
        
        for i in range(num_portfolios):
            pesos = np.random.random(num_ativos)
            pesos /= np.sum(pesos)
            
            ret = retorno_portfolio(pesos, retorno_esperado)
            vol = volatilidade_portfolio(pesos, matriz_cov)
            sharpe = (ret - taxa_livre_risco) / vol
            
            resultados[0, i] = ret
            resultados[1, i] = vol
            resultados[2, i] = sharpe
    
    fig_fronteira = go.Figure()
    
    fig_fronteira.add_trace(go.Scatter(
        x=resultados[1, :] * 100,
        y=resultados[0, :] * 100,
        mode='markers',
        marker=dict(size=3, color=resultados[2, :], colorscale='Viridis', showscale=True),
        name='Simulações'
    ))
    
    fig_fronteira.add_trace(go.Scatter(
        x=[vol_sharpe * 100],
        y=[ret_sharpe * 100],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Máx Sharpe'
    ))
    
    fig_fronteira.add_trace(go.Scatter(
        x=[vol_min_vol * 100],
        y=[ret_min_vol * 100],
        mode='markers',
        marker=dict(size=15, color='green', symbol='diamond'),
        name='Mín Vol'
    ))
    
    fig_fronteira.update_layout(
        title="Fronteira Eficiente",
        xaxis_title="Volatilidade (%)",
        yaxis_title="Retorno (%)",
        height=600
    )
    
    st.plotly_chart(fig_fronteira, use_container_width=True)
    
    # Performance
    st.markdown("---")
    st.subheader("📊 Performance Histórica")
    
    retornos_sharpe = (retornos * pesos_sharpe).sum(axis=1)
    retornos_min_vol = (retornos * pesos_min_vol).sum(axis=1)
    
    valor_sharpe = capital_inicial * (1 + retornos_sharpe).cumprod()
    valor_min_vol = capital_inicial * (1 + retornos_min_vol).cumprod()
    
    fig_perf = go.Figure()
    
    fig_perf.add_trace(go.Scatter(
        x=valor_sharpe.index,
        y=valor_sharpe.values,
        mode='lines',
        name='Máx Sharpe',
        line=dict(color='red', width=2)
    ))
    
    fig_perf.add_trace(go.Scatter(
        x=valor_min_vol.index,
        y=valor_min_vol.values,
        mode='lines',
        name='Mín Vol',
        line=dict(color='green', width=2)
    ))
    
    fig_perf.update_layout(
        title="Evolução do Valor",
        xaxis_title="Data",
        yaxis_title="Valor (R$)",
        height=500
    )
    
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # Resumo
    st.markdown("---")
    st.subheader("📋 Resumo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Valor Final (Máx Sharpe)",
            f"R$ {valor_sharpe.iloc[-1]:,.2f}",
            f"{((valor_sharpe.iloc[-1] / capital_inicial - 1) * 100):.2f}%"
        )
    
    with col2:
        st.metric(
            "Valor Final (Mín Vol)",
            f"R$ {valor_min_vol.iloc[-1]:,.2f}",
            f"{((valor_min_vol.iloc[-1] / capital_inicial - 1) * 100):.2f}%"
        )
    
    with col3:
        melhor = "Máx Sharpe" if valor_sharpe.iloc[-1] > valor_min_vol.iloc[-1] else "Mín Vol"
        st.metric("Melhor", melhor)
    
    # Downloads
    st.markdown("---")
    st.subheader("💾 Exportar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_sharpe = df_sharpe.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Máx Sharpe (CSV)",
            csv_sharpe,
            f"max_sharpe_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    
    with col2:
        csv_min = df_min_vol.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Mín Vol (CSV)",
            csv_min,
            f"min_vol_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

# Rodapé
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>📈 Otimização de Carteira | Dados: Yahoo Finance</p>
        <p style='font-size: 0.8em;'>⚠️ Apenas para fins educacionais</p>
    </div>
""", unsafe_allow_html=True)
