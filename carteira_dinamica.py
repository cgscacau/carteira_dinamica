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
            title="Retorno Médio Diário dos Ativos",
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
            title="Volatilidade Anualizada dos Ativos",
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
        title="Matriz de Correlação entre Ativos",
        height=600,
        xaxis={'side': 'bottom'}
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # ========== EVOLUÇÃO DOS PREÇOS ==========
    st.subheader("📈 Evolução Histórica dos Preços")
    
    # Normalizar preços para base 100
    dados_norm = (dados / dados.iloc[0] * 100)
    
    fig_precos = go.Figure()
    
    for ativo in dados_norm.columns:
        fig_precos.add_trace(go.Scatter(
            x=dados_norm.index,
            y=dados_norm[ativo],
            mode='lines',
            name=ativo,
            hovertemplate='<b>%{fullData.name}</b><br>%{y:.2f}<extra></extra>'
        ))
    
    fig_precos.update_layout(
        title="Evolução dos Preços (Base 100)",
        xaxis_title="Data",
        yaxis_title="Valor (Base 100)",
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
    
    with st.spinner("🔄 Otimizando carteiras..."):
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
    
    # ========== COMPARAÇÃO DAS ESTRATÉGIAS ==========
    st.subheader("📊 Comparação das Estratégias")
    
    estrategias = []
    cores = ['#FF4B4B', '#00CC00', '#FFD700']
    
    estrategias.append({
        'Nome': '🏆 Máximo Sharpe',
        'Descrição': 'Melhor retorno ajustado ao risco',
        'Retorno': ret_sharpe * 100,
        'Volatilidade': vol_sharpe * 100,
        'Sharpe': sharpe_sharpe,
        'Pesos': pesos_sharpe,
        'Cor': cores[0]
    })
    
    estrategias.append({
        'Nome': '🛡️ Mínima Volatilidade',
        'Descrição': 'Menor risco possível',
        'Retorno': ret_min_vol * 100,
        'Volatilidade': vol_min_vol * 100,
        'Sharpe': sharpe_min_vol,
        'Pesos': pesos_min_vol,
        'Cor': cores[1]
    })
    
    if pesos_div is not None:
        estrategias.append({
            'Nome': '💰 Foco Dividendos',
            'Descrição': 'Máxima renda passiva',
            'Retorno': ret_div * 100,
            'Volatilidade': vol_div * 100,
            'Sharpe': sharpe_div,
            'Pesos': pesos_div,
            'Cor': cores[2]
        })
    
    # Tabela Comparativa
    df_comp = pd.DataFrame([{
        'Estratégia': e['Nome'],
        'Descrição': e['Descrição'],
        'Retorno Anual': f"{e['Retorno']:.2f}%",
        'Volatilidade': f"{e['Volatilidade']:.2f}%",
        'Sharpe Ratio': f"{e['Sharpe']:.2f}"
    } for e in estrategias])
    
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    # ========== GRÁFICO COMPARATIVO DE MÉTRICAS ==========
    st.subheader("📊 Comparação Visual das Métricas")
    
    fig_comp = go.Figure()
    
    metricas = ['Retorno (%)', 'Sharpe Ratio', 'Volatilidade (%)']
    
    for estrategia in estrategias:
        fig_comp.add_trace(go.Bar(
            name=estrategia['Nome'],
            x=metricas,
            y=[estrategia['Retorno'], estrategia['Sharpe'] * 10, estrategia['Volatilidade']],
            marker_color=estrategia['Cor'],
            text=[f"{estrategia['Retorno']:.2f}%", 
                  f"{estrategia['Sharpe']:.2f}", 
                  f"{estrategia['Volatilidade']:.2f}%"],
            textposition='outside'
        ))
    
    fig_comp.update_layout(
        title="Comparação de Métricas (Sharpe x10 para visualização)",
        xaxis_title="Métrica",
        yaxis_title="Valor",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # ========== RECOMENDAÇÃO ==========
    st.subheader("🎖️ Melhor Estratégia por Critério")
    
    melhor_sharpe = max(estrategias, key=lambda x: x['Sharpe'])
    menor_risco = min(estrategias, key=lambda x: x['Volatilidade'])
    maior_retorno = max(estrategias, key=lambda x: x['Retorno'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"""
        **🏆 Melhor Sharpe**
        
        {melhor_sharpe['Nome']}
        
        Sharpe: **{melhor_sharpe['Sharpe']:.2f}**
        """)
    
    with col2:
        st.info(f"""
        **🛡️ Menor Risco**
        
        {menor_risco['Nome']}
        
        Vol: **{menor_risco['Volatilidade']:.2f}%**
        """)
    
    with col3:
        st.warning(f"""
        **📈 Maior Retorno**
        
        {maior_retorno['Nome']}
        
        Ret: **{maior_retorno['Retorno']:.2f}%**
        """)
    
    # ========== PERFIS DE INVESTIDOR ==========
    st.subheader("👤 Recomendação por Perfil")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        ### 🛡️ CONSERVADOR
        
        **Recomendação:** {menor_risco['Nome']}
        
        - Menor volatilidade
        - Preservação de capital
        - Risco minimizado
        
        **Métricas:**
        - Retorno: {menor_risco['Retorno']:.2f}%
        - Risco: {menor_risco['Volatilidade']:.2f}%
        - Sharpe: {menor_risco['Sharpe']:.2f}
        """)
    
    with col2:
        st.markdown(f"""
        ### ⚖️ MODERADO
        
        **Recomendação:** {melhor_sharpe['Nome']}
        
        - Melhor relação risco/retorno
        - Eficiência otimizada
        - Equilíbrio ideal
        
        **Métricas:**
        - Retorno: {melhor_sharpe['Retorno']:.2f}%
        - Risco: {melhor_sharpe['Volatilidade']:.2f}%
        - Sharpe: {melhor_sharpe['Sharpe']:.2f}
        """)
    
    with col3:
        if pesos_div is not None:
            estrategia_renda = [e for e in estrategias if '💰' in e['Nome']][0]
            st.markdown(f"""
            ### 💰 RENDA PASSIVA
            
            **Recomendação:** {estrategia_renda['Nome']}
            
            - Foco em dividendos
            - Renda mensal regular
            - Fluxo de caixa
            
            **Métricas:**
            - Retorno: {estrategia_renda['Retorno']:.2f}%
            - Risco: {estrategia_renda['Volatilidade']:.2f}%
            - Sharpe: {estrategia_renda['Sharpe']:.2f}
            """)
        else:
            st.markdown(f"""
            ### 🚀 AGRESSIVO
            
            **Recomendação:** {maior_retorno['Nome']}
            
            - Máximo retorno
            - Crescimento acelerado
            - Aceita volatilidade
            
            **Métricas:**
            - Retorno: {maior_retorno['Retorno']:.2f}%
            - Risco: {maior_retorno['Volatilidade']:.2f}%
            - Sharpe: {maior_retorno['Sharpe']:.2f}
            """)
    
    # ========== ALOCAÇÃO DAS CARTEIRAS ==========
    st.divider()
    st.header("📋 Alocação das Carteiras")
    
    tabs = st.tabs([e['Nome'] for e in estrategias])
    
    for tab, estrategia in zip(tabs, estrategias):
        with tab:
            col1, col2 = st.columns([3, 2])
            
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
                    }).background_gradient(subset=['Peso (%)'], cmap='Greens'),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Métricas
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("📈 Retorno Anual", f"{estrategia['Retorno']:.2f}%")
                with col_b:
                    st.metric("📊 Volatilidade", f"{estrategia['Volatilidade']:.2f}%")
                with col_c:
                    st.metric("⚡ Sharpe", f"{estrategia['Sharpe']:.2f}")
                with col_d:
                    valor_final = capital_inicial * (1 + estrategia['Retorno'] / 100)
                    st.metric("💰 Valor Final (1 ano)", f"R$ {valor_final:,.2f}")
            
            with col2:
                # Gráfico de Pizza
                fig_pie = go.Figure(data=[go.Pie(
                    labels=df_alocacao['Ativo'],
                    values=df_alocacao['Peso (%)'],
                    hole=0.4,
                    marker=dict(colors=[estrategia['Cor']] * len(df_alocacao)),
                    textinfo='label+percent',
                    textposition='auto'
                )])
                
                fig_pie.update_layout(
                    title=f"Distribuição - {estrategia['Nome']}",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
    
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
            size=5,
            color=resultados[2, :],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe<br>Ratio")
        ),
        name='Carteiras Possíveis',
        hovertemplate='Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>'
    ))
    
    # Carteiras ótimas
    for estrategia in estrategias:
        fig_front.add_trace(go.Scatter(
            x=[estrategia['Volatilidade']],
            y=[estrategia['Retorno']],
            mode='markers+text',
            marker=dict(
                size=25,
                color=estrategia['Cor'],
                symbol='star',
                line=dict(width=2, color='white')
            ),
            text=[estrategia['Nome']],
            textposition='top center',
            name=estrategia['Nome'],
            hovertemplate=f"<b>{estrategia['Nome']}</b><br>" +
                         f"Retorno: {estrategia['Retorno']:.2f}%<br>" +
                         f"Volatilidade: {estrategia['Volatilidade']:.2f}%<br>" +
                         f"Sharpe: {estrategia['Sharpe']:.2f}<extra></extra>"
        ))
    
    fig_front.update_layout(
        title="Fronteira Eficiente - Todas as Carteiras Possíveis",
        xaxis_title="Volatilidade Anual (%)",
        yaxis_title="Retorno Anual (%)",
        height=700,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
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
            name=estrategia['Nome'],
            line=dict(width=3, color=estrategia['Cor']),
            hovertemplate='<b>%{fullData.name}</b><br>R$ %{y:,.2f}<extra></extra>'
        ))
    
    # Linha de capital inicial
    fig_perf.add_hline(
        y=capital_inicial,
        line_dash="dash",
        line_color="gray",
        annotation_text="Capital Inicial",
        annotation_position="right"
    )
    
    fig_perf.update_layout(
        title="Evolução do Valor da Carteira ao Longo do Tempo",
        xaxis_title="Data",
        yaxis_title="Valor da Carteira (R$)",
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # Estatísticas da simulação
    st.subheader("📊 Estatísticas da Simulação")
    
    cols = st.columns(len(estrategias))
    
    for col, estrategia in zip(cols, estrategias):
        with col:
            retornos_estrategia = (retornos * estrategia['Pesos']).sum(axis=1)
            valor_estrategia = capital_inicial * (1 + retornos_estrategia).cumprod()
            
            valor_final = valor_estrategia.iloc[-1]
            retorno_total = ((valor_final / capital_inicial) - 1) * 100
            max_drawdown = ((valor_estrategia / valor_estrategia.cummax()) - 1).min() * 100
            
            st.markdown(f"""
            **{estrategia['Nome']}**
            
            - Valor Final: R$ {valor_final:,.2f}
            - Retorno Total: {retorno_total:.2f}%
            - Max Drawdown: {max_drawdown:.2f}%
            """)
    
    # ========== ANÁLISE DE DIVIDENDOS ==========
    if df_dividendos is not None and not df_dividendos.empty:
        st.divider()
        st.header("💰 Análise Detalhada de Dividendos")
        
        # Gráfico de barras empilhadas
        fig_div = go.Figure()
        
        for ativo in df_dividendos.columns:
            fig_div.add_trace(go.Bar(
                name=ativo,
                x=df_dividendos.index.strftime('%b/%y'),
                y=df_dividendos[ativo],
                hovertemplate='<b>%{fullData.name}</b><br>R$ %{y:.2f}<extra></extra>'
            ))
        
        fig_div.update_layout(
            title="Distribuição Mensal de Dividendos por Ativo",
            xaxis_title="Mês",
            yaxis_title="Dividendos (R$)",
            barmode='stack',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_div, use_container_width=True)
        
        # Gráfico de dividendos acumulados
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
            title="Evolução dos Dividendos Acumulados",
            xaxis_title="Mês",
            yaxis_title="Dividendos Acumulados (R$)",
            height=450,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_div_acum, use_container_width=True)
        
        # Métricas de dividendos
        st.subheader("📊 Métricas de Dividendos")
        
        total = df_dividendos.sum().sum()
        media = df_dividendos.sum(axis=1).mean()
        projecao = media * 12
        meses_pagantes = (df_dividendos.sum(axis=1) > 0).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Total Recebido", f"R$ {total:,.2f}")
        with col2:
            st.metric("📅 Média Mensal", f"R$ {media:,.2f}")
        with col3:
            st.metric("📈 Projeção Anual", f"R$ {projecao:,.2f}")
        with col4:
            st.metric("✅ Meses Pagantes", f"{meses_pagantes}/{len(df_dividendos)}")
        
        # Ranking de dividendos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Ranking de Dividendos")
            div_total = df_dividendos.sum().sort_values(ascending=False)
            
            fig_rank = go.Figure(go.Bar(
                x=div_total.values,
                y=div_total.index,
                orientation='h',
                marker=dict(color='gold'),
                text=div_total.apply(lambda x: f'R$ {x:.2f}'),
                textposition='outside'
            ))
            
            fig_rank.update_layout(
                title="Total de Dividendos por Ativo",
                xaxis_title="Dividendos (R$)",
                yaxis_title="Ativo",
                height=400
            )
            
            st.plotly_chart(fig_rank, use_container_width=True)
        
        with col2:
            st.subheader("📊 Dividend Yield")
            
            dy_data = []
            for ativo in df_dividendos.columns:
                preco_medio = dados[ativo].mean()
                div_anual = (df_dividendos[ativo].sum() / len(df_dividendos)) * 12
                dy = (div_anual / preco_medio) * 100 if preco_medio > 0 else 0
                dy_data.append({'Ativo': ativo, 'DY': dy})
            
            df_dy = pd.DataFrame(dy_data).sort_values('DY', ascending=False)
            
            fig_dy = go.Figure(go.Bar(
                x=df_dy['DY'],
                y=df_dy['Ativo'],
                orientation='h',
                marker=dict(color='lightgreen'),
                text=df_dy['DY'].apply(lambda x: f'{x:.2f}%'),
                textposition='outside'
            ))
            
            fig_dy.update_layout(
                title="Dividend Yield Anualizado",
                xaxis_title="DY (%)",
                yaxis_title="Ativo",
                height=400
            )
            
            st.plotly_chart(fig_dy, use_container_width=True)
    
    # ========== DOWNLOAD DOS RESULTADOS ==========
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
            nome = estrategia['Nome'].replace('🏆 ', '').replace('🛡️ ', '').replace('💰 ', '')
            nome = nome.replace(' ', '_').lower()
            
            st.download_button(
                label=f"📥 {estrategia['Nome']}",
                data=csv,
                file_name=f"{nome}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==================== RODAPÉ ==========
st.divider()
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <p style='font-size: 1rem;'>
        📊 <b>Otimizador de Carteira - Teoria Moderna de Portfólio (Markowitz)</b><br>
        Dados: Yahoo Finance | Desenvolvido com Streamlit<br>
        ⚠️ Apenas para fins educacionais - Não constitui recomendação de investimento
    </p>
</div>
""", unsafe_allow_html=True)
