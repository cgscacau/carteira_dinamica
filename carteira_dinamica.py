import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime, timedelta

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

# Seção de seleção de ativos
st.subheader("📊 Seleção de Ativos")

# Lista de ativos pré-definidos para busca rápida
ativos_populares = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
    'WEGE3.SA', 'RENT3.SA', 'MGLU3.SA', 'B3SA3.SA', 'SUZB3.SA',
    'BBAS3.SA', 'SANB11.SA', 'EGIE3.SA', 'CPLE6.SA', 'VIVT3.SA',
    'RADL3.SA', 'RAIL3.SA', 'EMBR3.SA', 'CSNA3.SA', 'USIM5.SA'
]

# Criar abas para diferentes métodos de seleção
tab1, tab2 = st.tabs(["🔍 Buscar Ativos", "✏️ Adicionar Manualmente"])

ativos_selecionados_busca = []
ativos_manuais = []

with tab1:
    st.write("Selecione ativos da lista de ações populares da B3:")
    ativos_selecionados_busca = st.multiselect(
        "Ativos disponíveis:",
        options=sorted(ativos_populares),
        default=['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA'],
        help="Selecione os ativos que deseja incluir na análise"
    )

with tab2:
    st.write("Adicione ativos manualmente digitando os tickers:")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("💡 **Dica:** Para ações brasileiras, adicione '.SA' ao final (ex: PETR4.SA). Para ações americanas, use apenas o ticker (ex: AAPL, MSFT)")
    
    with col2:
        adicionar_sufixo = st.checkbox(
            "Auto .SA",
            value=False,
            help="Adicionar '.SA' automaticamente"
        )
    
    # Campo para adicionar múltiplos ativos
    ativos_manuais_input = st.text_area(
        "Digite os tickers (separados por vírgula, espaço ou quebra de linha):",
        placeholder="Exemplo:\nPETR4.SA, VALE3.SA, ITUB4.SA\n\nou\n\nAAPL\nMSFT\nGOOGL",
        height=120,
        help="Você pode digitar múltiplos ativos separados por vírgula, espaço ou quebra de linha"
    )
    
    # Processar ativos manuais
    if ativos_manuais_input:
        # Substituir vírgulas e quebras de linha por espaços e dividir
        ativos_temp = ativos_manuais_input.replace(',', ' ').replace('\n', ' ').split()
        # Limpar espaços extras e converter para maiúsculas
        ativos_manuais = [ativo.strip().upper() for ativo in ativos_temp if ativo.strip()]
        
        # Adicionar sufixo se solicitado
        if adicionar_sufixo:
            ativos_manuais = [
                ativo if '.SA' in ativo.upper() or '.' in ativo else f"{ativo}.SA" 
                for ativo in ativos_manuais
            ]
        
        # Remover duplicatas
        ativos_manuais = list(set(ativos_manuais))
        
        if ativos_manuais:
            st.success(f"✅ {len(ativos_manuais)} ativo(s) adicionado(s): {', '.join(ativos_manuais)}")
            
            # Validar ativos
            if st.checkbox("🔍 Validar ativos antes de usar", value=True):
                with st.spinner("Validando ativos..."):
                    ativos_validos = []
                    ativos_invalidos = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, ativo in enumerate(ativos_manuais):
                        status_text.text(f"Validando {ativo}...")
                        try:
                            ticker = yf.Ticker(ativo)
                            # Tentar obter dados recentes
                            hist = ticker.history(period="5d")
                            if not hist.empty and len(hist) > 0:
                                ativos_validos.append(ativo)
                            else:
                                ativos_invalidos.append(ativo)
                        except Exception as e:
                            ativos_invalidos.append(ativo)
                        
                        progress_bar.progress((i + 1) / len(ativos_manuais))
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if ativos_validos:
                        st.success(f"✅ Ativos válidos ({len(ativos_validos)}): {', '.join(ativos_validos)}")
                    
                    if ativos_invalidos:
                        st.error(f"❌ Ativos inválidos ou sem dados ({len(ativos_invalidos)}): {', '.join(ativos_invalidos)}")
                        st.warning("⚠️ Os ativos inválidos serão removidos da análise.")
                    
                    # Atualizar lista com apenas ativos válidos
                    ativos_manuais = ativos_validos

# Combinar ativos de ambas as fontes e remover duplicatas
ativos_finais = list(set(ativos_selecionados_busca + ativos_manuais))

# Mostrar resumo final
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Ativos da Busca", len(ativos_selecionados_busca))
with col2:
    st.metric("Ativos Manuais", len(ativos_manuais))
with col3:
    st.metric("Total de Ativos", len(ativos_finais))

if ativos_finais:
    with st.expander("📋 Ver lista completa de ativos selecionados", expanded=False):
        # Organizar em colunas para melhor visualização
        num_cols = 4
        cols = st.columns(num_cols)
        for i, ativo in enumerate(sorted(ativos_finais)):
            with cols[i % num_cols]:
                st.write(f"• {ativo}")
else:
    st.warning("⚠️ Nenhum ativo selecionado. Por favor, selecione ou adicione ativos para continuar.")
    st.stop()

# Verificar se o período é válido
if data_inicio >= data_fim:
    st.error("❌ A data inicial deve ser anterior à data final!")
    st.stop()

# Botão para iniciar análise
st.markdown("---")
if st.button("🚀 Iniciar Análise e Otimização", type="primary", use_container_width=True):
    
# Botão para iniciar análise
st.markdown("---")
if st.button("🚀 Iniciar Análise e Otimização", type="primary", use_container_width=True):
    
    # Baixar dados
    st.subheader("📥 Baixando Dados dos Ativos")
    
    with st.spinner("Baixando dados históricos..."):
        try:
            # Tentar baixar dados
            if len(ativos_finais) == 1:
                # Para um único ativo
                ticker = yf.Ticker(ativos_finais[0])
                dados_temp = ticker.history(start=data_inicio, end=data_fim)
                if dados_temp.empty:
                    st.error(f"❌ Não foi possível obter dados para {ativos_finais[0]}")
                    st.stop()
                dados = pd.DataFrame({ativos_finais[0]: dados_temp['Close']})
            else:
                # Para múltiplos ativos
                dados = yf.download(
                    ativos_finais,
                    start=data_inicio,
                    end=data_fim,
                    progress=False
                )
                
                # Verificar se retornou dados
                if dados.empty:
                    st.error("❌ Não foi possível obter dados para os ativos selecionados.")
                    st.stop()
                
                # Tentar acessar 'Adj Close', se não existir usar 'Close'
                if 'Adj Close' in dados.columns.get_level_values(0):
                    dados = dados['Adj Close']
                elif 'Close' in dados.columns.get_level_values(0):
                    dados = dados['Close']
                else:
                    # Se os dados vieram em formato diferente
                    if isinstance(dados.columns, pd.MultiIndex):
                        dados = dados.xs('Close', level=0, axis=1)
                    else:
                        # Dados já estão no formato correto
                        pass
            
            # Garantir que dados seja um DataFrame
            if not isinstance(dados, pd.DataFrame):
                dados = dados.to_frame()
            
            # Remover colunas completamente vazias
            dados = dados.dropna(axis=1, how='all')
            
            # Remover linhas com muitos NaN (mais de 50% de NaN)
            threshold = len(dados.columns) * 0.5
            dados = dados.dropna(thresh=threshold)
            
            # Preencher NaN restantes com forward fill e depois backward fill
            dados = dados.fillna(method='ffill').fillna(method='bfill')
            
            if dados.empty:
                st.error("❌ Não foi possível obter dados válidos para os ativos selecionados no período especificado.")
                st.info("💡 **Dicas:**\n- Verifique se os tickers estão corretos\n- Tente um período de datas diferente\n- Verifique se os ativos têm histórico de negociação no período selecionado")
                st.stop()
            
            # Atualizar lista de ativos com apenas os que têm dados
            ativos_com_dados = dados.columns.tolist()
            
            if len(ativos_com_dados) < len(ativos_finais):
                ativos_sem_dados = set(ativos_finais) - set(ativos_com_dados)
                st.warning(f"⚠️ Os seguintes ativos não possuem dados no período selecionado e foram removidos: {', '.join(ativos_sem_dados)}")
            
            if len(ativos_com_dados) < 2:
                st.error("❌ É necessário pelo menos 2 ativos com dados válidos para otimização de carteira.")
                st.stop()
            
            st.success(f"✅ Dados baixados com sucesso para {len(ativos_com_dados)} ativos!")
            
            # Mostrar preview dos dados
            with st.expander("👁️ Visualizar dados históricos", expanded=False):
                st.write(f"**Período:** {dados.index[0].strftime('%d/%m/%Y')} até {dados.index[-1].strftime('%d/%m/%Y')}")
                st.write(f"**Total de dias:** {len(dados)}")
                st.dataframe(dados.tail(10).style.format('{:.2f}'), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Erro ao baixar dados: {str(e)}")
            st.info("""
            **Possíveis causas:**
            - Tickers inválidos ou incorretos
            - Problemas de conexão com Yahoo Finance
            - Período de datas sem dados disponíveis
            - Ativos deslistados ou sem histórico
            
            **Sugestões:**
            - Verifique se os tickers estão corretos (ex: PETR4.SA para ações brasileiras)
            - Tente com ativos diferentes
            - Verifique sua conexão com a internet
            - Tente um período de datas mais recente
            """)
            st.stop()

    
    # Calcular retornos
    st.subheader("📊 Análise de Retornos")
    
    retornos = dados.pct_change().dropna()
    
    # Estatísticas dos ativos
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Retorno Médio Diário (%)**")
        retorno_medio = (retornos.mean() * 100).sort_values(ascending=False)
        st.dataframe(retorno_medio.to_frame('Retorno (%)'), use_container_width=True)
    
    with col2:
        st.write("**Volatilidade (Desvio Padrão) (%)**")
        volatilidade = (retornos.std() * 100).sort_values(ascending=False)
        st.dataframe(volatilidade.to_frame('Volatilidade (%)'), use_container_width=True)
    
    # Matriz de correlação
    st.write("**Matriz de Correlação entre Ativos**")
    correlacao = retornos.corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlacao.values,
        x=correlacao.columns,
        y=correlacao.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlacao.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlação")
    ))
    
    fig_corr.update_layout(
        title="Matriz de Correlação",
        xaxis_title="Ativos",
        yaxis_title="Ativos",
        height=600
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Otimização da carteira
    st.subheader("🎯 Otimização da Carteira")
    
    # Calcular retorno esperado e matriz de covariância
    retorno_esperado = retornos.mean()
    matriz_cov = retornos.cov()
    
    # Número de ativos
    num_ativos = len(ativos_com_dados)
    
    # Função objetivo: minimizar volatilidade (risco)
    def volatilidade_portfolio(pesos, matriz_cov):
        return np.sqrt(np.dot(pesos.T, np.dot(matriz_cov, pesos))) * np.sqrt(252)
    
    # Função para calcular retorno do portfolio
    def retorno_portfolio(pesos, retorno_esperado):
        return np.sum(retorno_esperado * pesos) * 252
    
    # Função para calcular Sharpe Ratio
    def sharpe_ratio_negativo(pesos, retorno_esperado, matriz_cov, taxa_livre_risco):
        ret = retorno_portfolio(pesos, retorno_esperado)
        vol = volatilidade_portfolio(pesos, matriz_cov)
        return -(ret - taxa_livre_risco) / vol
    
    # Restrições e limites
    restricoes = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    limites = tuple((0, 1) for _ in range(num_ativos))
    pesos_iniciais = np.array([1/num_ativos] * num_ativos)
    
    # Otimização 1: Máximo Sharpe Ratio
    with st.spinner("Otimizando carteira para máximo Sharpe Ratio..."):
        resultado_sharpe = minimize(
            sharpe_ratio_negativo,
            pesos_iniciais,
            args=(retorno_esperado, matriz_cov, taxa_livre_risco),
            method='SLSQP',
            bounds=limites,
            constraints=restricoes
        )
    
    # Otimização 2: Mínima Volatilidade
    with st.spinner("Otimizando carteira para mínima volatilidade..."):
        resultado_min_vol = minimize(
            volatilidade_portfolio,
            pesos_iniciais,
            args=(matriz_cov,),
            method='SLSQP',
            bounds=limites,
            constraints=restricoes
        )
    
    # Resultados
    pesos_sharpe = resultado_sharpe.x
    pesos_min_vol = resultado_min_vol.x
    
    # Calcular métricas para ambas as carteiras
    ret_sharpe = retorno_portfolio(pesos_sharpe, retorno_esperado)
    vol_sharpe = volatilidade_portfolio(pesos_sharpe, matriz_cov)
    sharpe_sharpe = (ret_sharpe - taxa_livre_risco) / vol_sharpe
    
    ret_min_vol = retorno_portfolio(pesos_min_vol, retorno_esperado)
    vol_min_vol = volatilidade_portfolio(pesos_min_vol, matriz_cov)
    sharpe_min_vol = (ret_min_vol - taxa_livre_risco) / vol_min_vol
    
    # Exibir resultados
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🏆 Carteira com Máximo Sharpe Ratio")
        st.metric("Retorno Esperado Anual", f"{ret_sharpe*100:.2f}%")
        st.metric("Volatilidade Anual", f"{vol_sharpe*100:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe_sharpe:.2f}")
        
        st.write("**Alocação:**")
        df_sharpe = pd.DataFrame({
            'Ativo': ativos_com_dados,
            'Peso (%)': pesos_sharpe * 100,
            'Valor (R$)': pesos_sharpe * capital_inicial
        })
        df_sharpe = df_sharpe[df_sharpe['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=False)
        st.dataframe(df_sharpe.style.format({
            'Peso (%)': '{:.2f}%',
            'Valor (R$)': 'R$ {:.2f}'
        }), use_container_width=True)
        
        # Gráfico de pizza
        fig_pie_sharpe = go.Figure(data=[go.Pie(
            labels=df_sharpe['Ativo'],
            values=df_sharpe['Peso (%)'],
            hole=0.3,
            textinfo='label+percent',
            textposition='auto'
        )])
        fig_pie_sharpe.update_layout(title="Distribuição da Carteira (Máximo Sharpe)")
        st.plotly_chart(fig_pie_sharpe, use_container_width=True)
    
    with col2:
        st.write("### 🛡️ Carteira de Mínima Volatilidade")
        st.metric("Retorno Esperado Anual", f"{ret_min_vol*100:.2f}%")
        st.metric("Volatilidade Anual", f"{vol_min_vol*100:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe_min_vol:.2f}")
        
        st.write("**Alocação:**")
        df_min_vol = pd.DataFrame({
            'Ativo': ativos_com_dados,
            'Peso (%)': pesos_min_vol * 100,
            'Valor (R$)': pesos_min_vol * capital_inicial
        })
        df_min_vol = df_min_vol[df_min_vol['Peso (%)'] > 0.01].sort_values('Peso (%)', ascending=False)
        st.dataframe(df_min_vol.style.format({
            'Peso (%)': '{:.2f}%',
            'Valor (R$)': 'R$ {:.2f}'
        }), use_container_width=True)
        
        # Gráfico de pizza
        fig_pie_min = go.Figure(data=[go.Pie(
            labels=df_min_vol['Ativo'],
            values=df_min_vol['Peso (%)'],
            hole=0.3,
            textinfo='label+percent',
            textposition='auto'
        )])
        fig_pie_min.update_layout(title="Distribuição da Carteira (Mínima Volatilidade)")
        st.plotly_chart(fig_pie_min, use_container_width=True)
    
    # Fronteira Eficiente
    st.markdown("---")
    st.subheader("📈 Fronteira Eficiente")
    
    with st.spinner("Calculando fronteira eficiente..."):
        # Gerar carteiras aleatórias
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
    
    # Plotar fronteira eficiente
    fig_fronteira = go.Figure()
    
    # Carteiras aleatórias
    fig_fronteira.add_trace(go.Scatter(
        x=resultados[1, :] * 100,
        y=resultados[0, :] * 100,
        mode='markers',
        marker=dict(
            size=3,
            color=resultados[2, :],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe<br>Ratio")
        ),
        name='Carteiras Simuladas',
        text=[f'Sharpe: {s:.2f}' for s in resultados[2, :]],
        hovertemplate='Volatilidade: %{x:.2f}%<br>Retorno: %{y:.2f}%<br>%{text}<extra></extra>'
    ))
    
    # Carteira de máximo Sharpe
    fig_fronteira.add_trace(go.Scatter(
        x=[vol_sharpe * 100],
        y=[ret_sharpe * 100],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Máximo Sharpe Ratio',
        hovertemplate='Volatilidade: %{x:.2f}%<br>Retorno: %{y:.2f}%<br>Sharpe: ' + f'{sharpe_sharpe:.2f}<extra></extra>'
    ))
    
    # Carteira de mínima volatilidade
    fig_fronteira.add_trace(go.Scatter(
        x=[vol_min_vol * 100],
        y=[ret_min_vol * 100],
        mode='markers',
        marker=dict(size=15, color='green', symbol='diamond'),
        name='Mínima Volatilidade',
        hovertemplate='Volatilidade: %{x:.2f}%<br>Retorno: %{y:.2f}%<br>Sharpe: ' + f'{sharpe_min_vol:.2f}<extra></extra>'
    ))
    
    fig_fronteira.update_layout(
        title="Fronteira Eficiente de Markowitz",
        xaxis_title="Volatilidade Anual (%)",
        yaxis_title="Retorno Esperado Anual (%)",
        height=600,
        hovermode='closest'
    )
    
    st.plotly_chart(fig_fronteira, use_container_width=True)
    
    # Simulação de performance histórica
    st.markdown("---")
    st.subheader("📊 Performance Histórica das Carteiras")
    
    # Calcular valor das carteiras ao longo do tempo
    retornos_sharpe = (retornos * pesos_sharpe).sum(axis=1)
    retornos_min_vol = (retornos * pesos_min_vol).sum(axis=1)
    
    valor_sharpe = capital_inicial * (1 + retornos_sharpe).cumprod()
    valor_min_vol = capital_inicial * (1 + retornos_min_vol).cumprod()
    
    # Gráfico de performance
    fig_performance = go.Figure()
    
    fig_performance.add_trace(go.Scatter(
        x=valor_sharpe.index,
        y=valor_sharpe.values,
        mode='lines',
        name='Máximo Sharpe Ratio',
        line=dict(color='red', width=2)
    ))
    
    fig_performance.add_trace(go.Scatter(
        x=valor_min_vol.index,
        y=valor_min_vol.values,
        mode='lines',
        name='Mínima Volatilidade',
        line=dict(color='green', width=2)
    ))
    
    fig_performance.update_layout(
        title="Evolução do Valor das Carteiras",
        xaxis_title="Data",
        yaxis_title="Valor da Carteira (R$)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_performance, use_container_width=True)
    
    # Estatísticas finais
    st.markdown("---")
    st.subheader("📋 Resumo Final")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Valor Final (Máx. Sharpe)",
            f"R$ {valor_sharpe.iloc[-1]:,.2f}",
            f"{((valor_sharpe.iloc[-1] / capital_inicial - 1) * 100):.2f}%"
        )
    
    with col2:
        st.metric(
            "Valor Final (Mín. Vol)",
            f"R$ {valor_min_vol.iloc[-1]:,.2f}",
            f"{((valor_min_vol.iloc[-1] / capital_inicial - 1) * 100):.2f}%"
        )
    
    with col3:
        melhor = "Máximo Sharpe" if valor_sharpe.iloc[-1] > valor_min_vol.iloc[-1] else "Mínima Volatilidade"
        st.metric("Melhor Performance", melhor)
    
    # Tabela comparativa final
    st.write("**Comparação Detalhada:**")
    
    df_comparacao = pd.DataFrame({
        'Métrica': [
            'Retorno Esperado Anual',
            'Volatilidade Anual',
            'Sharpe Ratio',
            'Valor Inicial',
            'Valor Final',
            'Retorno Total'
        ],
        'Máximo Sharpe Ratio': [
            f"{ret_sharpe*100:.2f}%",
            f"{vol_sharpe*100:.2f}%",
            f"{sharpe_sharpe:.2f}",
            f"R$ {capital_inicial:,.2f}",
            f"R$ {valor_sharpe.iloc[-1]:,.2f}",
            f"{((valor_sharpe.iloc[-1] / capital_inicial - 1) * 100):.2f}%"
        ],
        'Mínima Volatilidade': [
            f"{ret_min_vol*100:.2f}%",
            f"{vol_min_vol*100:.2f}%",
            f"{sharpe_min_vol:.2f}",
            f"R$ {capital_inicial:,.2f}",
            f"R$ {valor_min_vol.iloc[-1]:,.2f}",
            f"{((valor_min_vol.iloc[-1] / capital_inicial - 1) * 100):.2f}%"
        ]
    })
    
    st.dataframe(df_comparacao, use_container_width=True, hide_index=True)
    
    # Download dos resultados
    st.markdown("---")
    st.subheader("💾 Exportar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV da carteira Sharpe
        csv_sharpe = df_sharpe.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Carteira Máx. Sharpe (CSV)",
            data=csv_sharpe,
            file_name=f"carteira_max_sharpe_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # CSV da carteira Min Vol
        csv_min_vol = df_min_vol.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Carteira Mín. Vol (CSV)",
            data=csv_min_vol,
            file_name=f"carteira_min_vol_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Rodapé
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Desenvolvido com ❤️ usando Streamlit | Dados fornecidos por Yahoo Finance</p>
        <p style='font-size: 0.8em;'>⚠️ Este aplicativo é apenas para fins educacionais. 
        Não constitui aconselhamento financeiro. Consulte um profissional qualificado antes de investir.</p>
    </div>
""", unsafe_allow_html=True)
