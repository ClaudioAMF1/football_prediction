import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
from datetime import datetime

# Adicionar diretório src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_collector import BrasileiraoDataCollector
from src.data_processor import BrasileiraoDataProcessor
from src.model import BrasileiraoPredictor

# Configuração da página
st.set_page_config(
    page_title="Análise do Brasileirão",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            width: 100%;
            margin-top: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        .metric-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        .stPlotlyChart {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .st-emotion-cache-16txtl3 {
            padding: 1rem;
        }
        h1, h2, h3 {
            color: #1f1f1f;
            font-weight: 600;
        }
        .status-card {
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .success-status {
            background-color: #d4edda;
            color: #155724;
        }
        .error-status {
            background-color: #f8d7da;
            color: #721c24;
        }
    </style>
""", unsafe_allow_html=True)

# Inicializar session_state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False


# Funções de cache
@st.cache_resource
def load_resources():
    return BrasileiraoDataCollector(), BrasileiraoDataProcessor(), BrasileiraoPredictor()


@st.cache_data
def load_data():
    if os.path.exists('data/brasileirao_matches.csv'):
        df = pd.read_csv('data/brasileirao_matches.csv')
        df['data'] = pd.to_datetime(df['data'])
        return df
    return None


@st.cache_data
def load_standings():
    if os.path.exists('data/classificacao.csv'):
        return pd.read_csv('data/classificacao.csv')
    return None


# Carregar recursos
collector, processor, predictor = load_resources()

# Título principal
st.title("⚽ Análise e Previsão do Brasileirão 2024")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Controles")

    # Status
    st.markdown("#### 📊 Status do Sistema")
    data_exists = os.path.exists('data/brasileirao_matches.csv')
    model_exists = os.path.exists('models/brasileirao_predictor.joblib')

    status_color = "success-status" if data_exists else "error-status"
    st.markdown(f"""
        <div class="status-card {status_color}">
            {'✅' if data_exists else '❌'} Dados: {'Carregados' if data_exists else 'Não encontrados'}
        </div>
    """, unsafe_allow_html=True)

    status_color = "success-status" if model_exists else "error-status"
    st.markdown(f"""
        <div class="status-card {status_color}">
            {'✅' if model_exists else '❌'} Modelo: {'Treinado' if model_exists else 'Não treinado'}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Botões de ação
    st.markdown("#### 🔄 Ações")
    if st.button("📥 Atualizar Dados"):
        with st.spinner("Coletando dados do campeonato..."):
            try:
                df = collector.update_data()
                if df is not None:
                    st.session_state.data_loaded = True
                    st.success("✅ Dados atualizados com sucesso!")
                    st.rerun()
                else:
                    st.error("❌ Erro ao atualizar dados")
            except Exception as e:
                st.error(f"❌ Erro: {str(e)}")

    if st.button("🤖 Treinar Modelo"):
        if not os.path.exists('data/brasileirao_matches.csv'):
            st.error("❌ Atualize os dados primeiro!")
        else:
            with st.spinner("Treinando modelo..."):
                try:
                    df = load_data()
                    if df is not None:
                        X, y = processor.preparar_dados_treino(df)
                        if X is not None and y is not None and len(X) > 0:
                            results = predictor.treinar(X, y)
                            if 'error' in results:
                                st.error(f"❌ Erro: {results['error']}")
                            else:
                                predictor.salvar_modelo()
                                st.session_state.model_trained = True

                                st.success("✅ Modelo treinado com sucesso!")
                                st.markdown("##### Métricas")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Acurácia", f"{results['test_score']:.1%}")
                                with col2:
                                    st.metric("CV Score", f"{results['cv_mean']:.1%}")

                                with st.expander("📊 Relatório Detalhado"):
                                    st.text(results['classification_report'])
                        else:
                            st.error("❌ Dados insuficientes")
                    else:
                        st.error("❌ Erro ao carregar dados")
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)}")

    st.markdown("---")
    st.caption("Desenvolvido para análise do Brasileirão 2024")

# Interface principal - abas
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Classificação",
    "🎯 Previsões",
    "📈 Estatísticas",
    "🔍 Análise de Time"
])

# Tab Classificação
with tab1:
    st.header("📊 Classificação do Brasileirão 2024")
    standings = load_standings()

    if standings is not None:
        # Estilizar tabela
        def highlight_positions(val):
            if isinstance(val, int):
                if val <= 4:
                    return 'background-color: #90EE90; color: #000000'  # Libertadores
                elif val <= 6:
                    return 'background-color: #98FB98; color: #000000'  # Pré-Libertadores
                elif val <= 12:
                    return 'background-color: #87CEEB; color: #000000'  # Sul-Americana
                elif val >= 17:
                    return 'background-color: #FFB6C6; color: #000000'  # Rebaixamento
            return ''


        styled_standings = standings.style.applymap(
            highlight_positions,
            subset=['posicao']
        )

        # Container para a tabela
        with st.container():
            st.markdown("""
                <div class="metric-card">
                    <h4>Tabela de Classificação</h4>
                </div>
            """, unsafe_allow_html=True)
            st.dataframe(
                styled_standings,
                height=600,
                use_container_width=True
            )

        # Gráficos de análise
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de pontos
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=standings['time'],
                y=standings['pontos'],
                marker_color=px.colors.qualitative.Set3,
                text=standings['pontos'],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>" +
                              "Pontos: %{y}<br>" +
                              "<extra></extra>"
            ))

            fig.update_layout(
                title="Pontuação por Time",
                title_x=0.5,
                xaxis_title="",
                yaxis_title="Pontos",
                plot_bgcolor='white',
                showlegend=False,
                height=500,
                hoverlabel=dict(bgcolor="white"),
                xaxis_tickangle=-45
            )

            fig.update_xaxes(gridcolor='lightgrey')
            fig.update_yaxes(gridcolor='lightgrey')

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Gráfico de aproveitamento
            aproveitamento = (standings['pontos'] / (standings['jogos'] * 3) * 100).round(1)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=standings['time'],
                y=aproveitamento,
                marker_color=px.colors.sequential.Viridis,
                text=aproveitamento.apply(lambda x: f"{x:.1f}%"),
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>" +
                              "Aproveitamento: %{text}<br>" +
                              "<extra></extra>"
            ))

            fig.update_layout(
                title="Aproveitamento dos Times (%)",
                title_x=0.5,
                xaxis_title="",
                yaxis_title="Aproveitamento (%)",
                plot_bgcolor='white',
                showlegend=False,
                height=500,
                hoverlabel=dict(bgcolor="white"),
                xaxis_tickangle=-45
            )

            fig.update_xaxes(gridcolor='lightgrey')
            fig.update_yaxes(gridcolor='lightgrey')

            st.plotly_chart(fig, use_container_width=True)

        # Métricas do campeonato
        st.markdown("### 📈 Métricas do Campeonato")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            media_gols = (standings['gols_pro'].sum() / standings['jogos'].sum()).round(2)
            st.metric("Média de Gols/Jogo", f"{media_gols:.2f}")

        with col2:
            total_gols = standings['gols_pro'].sum()
            st.metric("Total de Gols", f"{total_gols}")

        with col3:
            melhor_ataque = standings.loc[standings['gols_pro'].idxmax()]
            st.metric("Melhor Ataque", f"{melhor_ataque['time']} ({melhor_ataque['gols_pro']})")

        with col4:
            melhor_defesa = standings.loc[standings['gols_contra'].idxmin()]
            st.metric("Melhor Defesa", f"{melhor_defesa['time']} ({melhor_defesa['gols_contra']})")
    else:
        st.error("❌ Dados da classificação não encontrados. Clique em 'Atualizar Dados'.")

# Tab Previsões
with tab2:
    st.header("🎯 Previsão de Partidas")

    if not st.session_state.model_trained:
        st.warning("⚠️ Modelo não treinado. Por favor, treine o modelo primeiro!")
    else:
        df = load_data()
        if df is not None:
            with st.container():
                st.markdown("""
                    <div class="metric-card">
                        <h4>Selecione os Times</h4>
                    </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    time_casa = st.selectbox(
                        "Time da Casa",
                        options=sorted(df['time_casa'].unique()),
                        key='home_team'
                    )

                with col2:
                    time_fora = st.selectbox(
                        "Time Visitante",
                        options=[t for t in sorted(df['time_fora'].unique()) if t != time_casa],
                        key='away_team'
                    )

                if st.button("🎲 Fazer Previsão", use_container_width=True):
                    try:
                        X = processor.preparar_dados_predicao(df, time_casa, time_fora)
                        if X is not None:
                            predictor.carregar_modelo()
                            probabilidades = predictor.prever_probabilidades(X)[0]

                            # Container para resultados
                            st.markdown("""
                                <div class="metric-card">
                                    <h4>Probabilidades</h4>
                                </div>
                            """, unsafe_allow_html=True)

                            # Probabilidades com cores
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; padding: 10px; background-color: #90EE90; border-radius: 5px;">
                                        <h4>Vitória {time_casa}</h4>
                                        <h2>{probabilidades[2]:.1%}</h2>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            with col2:
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; padding: 10px; background-color: #FFD700; border-radius: 5px;">
                                        <h4>Empate</h4>
                                        <h2>{probabilidades[1]:.1%}</h2>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            with col3:
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; padding: 10px; background-color: #87CEEB; border-radius: 5px;">
                                        <h4>Vitória {time_fora}</h4>
                                        <h2>{probabilidades[0]:.1%}</h2>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            # Gráfico de pizza com probabilidades
                            fig = go.Figure(data=[go.Pie(
                                labels=[f'Vitória {time_casa}', 'Empate', f'Vitória {time_fora}'],
                                values=probabilidades,
                                hole=.3,
                                marker_colors=['#90EE90', '#FFD700', '#87CEEB']
                            )])

                            fig.update_layout(
                                title="Distribuição de Probabilidades",
                                title_x=0.5,
                                height=400,
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Histórico Recente
                            st.markdown("### 📊 Forma Recente")
                            col1, col2 = st.columns(2)

                            with col1:
                                forma_casa = processor.obter_forma_recente(df, time_casa)
                                st.markdown(f"**{time_casa}**")
                                st.markdown(f"<h3>{''.join(forma_casa)}</h3>", unsafe_allow_html=True)

                            with col2:
                                forma_fora = processor.obter_forma_recente(df, time_fora)
                                st.markdown(f"**{time_fora}**")
                                st.markdown(f"<h3>{''.join(forma_fora)}</h3>", unsafe_allow_html=True)

                            # Confrontos diretos
                            st.markdown("### 🤝 Confrontos Diretos")
                            confrontos = df[
                                ((df['time_casa'] == time_casa) & (df['time_fora'] == time_fora)) |
                                ((df['time_casa'] == time_fora) & (df['time_fora'] == time_casa))
                                ].sort_values('data', ascending=False)

                            if not confrontos.empty:
                                for _, jogo in confrontos.head(5).iterrows():
                                    data = jogo['data'].strftime('%d/%m/%Y')
                                    if jogo['time_casa'] == time_casa:
                                        resultado = "✅" if jogo['vencedor'] == 'HOME_TEAM' else \
                                            "❌" if jogo['vencedor'] == 'AWAY_TEAM' else "➖"
                                    else:
                                        resultado = "✅" if jogo['vencedor'] == 'AWAY_TEAM' else \
                                            "❌" if jogo['vencedor'] == 'HOME_TEAM' else "➖"

                                    st.markdown(
                                        f"""
                                        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin: 5px 0;">
                                            {resultado} {data} - {jogo['time_casa']} {jogo['gols_casa']} x {jogo['gols_fora']} {jogo['time_fora']}
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.info("Nenhum confronto direto encontrado")
                        else:
                            st.error("❌ Dados insuficientes para previsão")
                    except Exception as e:
                        st.error(f"❌ Erro ao fazer previsão: {str(e)}")
        else:
            st.error("❌ Dados não encontrados")

# Tab Estatísticas
with tab3:
    st.header("📈 Estatísticas do Campeonato")

    df = load_data()
    if df is not None:
        try:
            # Estatísticas gerais em cards
            st.markdown("### 📊 Visão Geral")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_jogos = len(df[df['status'] == 'FINISHED'])
                st.markdown(
                    f"""
                    <div class="metric-card" style="text-align: center;">
                        <h4>Total de Jogos</h4>
                        <h2>{total_jogos}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                media_gols = (df['gols_casa'].mean() + df['gols_fora'].mean()) / 2
                st.markdown(
                    f"""
                    <div class="metric-card" style="text-align: center;">
                        <h4>Média de Gols/Jogo</h4>
                        <h2>{media_gols:.2f}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:
                vitorias_casa = len(df[df['vencedor'] == 'HOME_TEAM'])
                aproveitamento_casa = vitorias_casa / total_jogos * 100
                st.markdown(
                    f"""
                    <div class="metric-card" style="text-align: center;">
                        <h4>Vitórias em Casa</h4>
                        <h2>{aproveitamento_casa:.1f}%</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col4:
                empates = len(df[df['vencedor'] == 'DRAW'])
                taxa_empates = empates / total_jogos * 100
                st.markdown(
                    f"""
                    <div class="metric-card" style="text-align: center;">
                        <h4>Taxa de Empates</h4>
                        <h2>{taxa_empates:.1f}%</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Gráficos de análise
            st.markdown("### 📊 Análises Detalhadas")
            col1, col2 = st.columns(2)

            with col1:
                # Gráfico de gols por rodada
                gols_rodada = df.groupby('rodada').agg({
                    'gols_casa': 'sum',
                    'gols_fora': 'sum'
                }).reset_index()

                gols_rodada['total_gols'] = gols_rodada['gols_casa'] + gols_rodada['gols_fora']

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=gols_rodada['rodada'],
                    y=gols_rodada['total_gols'],
                    mode='lines+markers',
                    name='Total de Gols',
                    line=dict(color='#2E86C1', width=3),
                    marker=dict(size=8),
                    hovertemplate="Rodada %{x}<br>Gols: %{y}<extra></extra>"
                ))

                fig.update_layout(
                    title="Gols por Rodada",
                    title_x=0.5,
                    xaxis_title="Rodada",
                    yaxis_title="Número de Gols",
                    plot_bgcolor='white',
                    hoverlabel=dict(bgcolor="white"),
                    height=400
                )

                fig.update_xaxes(gridcolor='lightgrey', tickmode='linear')
                fig.update_yaxes(gridcolor='lightgrey')

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Distribuição de resultados
                resultados = df['vencedor'].value_counts()
                labels = {
                    'HOME_TEAM': 'Vitória Casa',
                    'AWAY_TEAM': 'Vitória Fora',
                    'DRAW': 'Empate'
                }
                resultados.index = resultados.index.map(labels)

                fig = go.Figure(data=[go.Pie(
                    labels=resultados.index,
                    values=resultados.values,
                    hole=.3,
                    marker_colors=['#2ECC71', '#3498DB', '#F1C40F']
                )])

                fig.update_layout(
                    title="Distribuição de Resultados",
                    title_x=0.5,
                    height=400,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig, use_container_width=True)

            # Artilharia por time
            st.markdown("### ⚽ Artilharia por Time")

            gols_times = pd.DataFrame()
            gols_pro = df.groupby('time_casa')['gols_casa'].sum() + \
                       df.groupby('time_fora')['gols_fora'].sum()

            gols_contra = df.groupby('time_casa')['gols_fora'].sum() + \
                          df.groupby('time_fora')['gols_casa'].sum()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Gols Marcados',
                x=gols_pro.index,
                y=gols_pro.values,
                marker_color='#2ECC71'
            ))

            fig.add_trace(go.Bar(
                name='Gols Sofridos',
                x=gols_contra.index,
                y=gols_contra.values,
                marker_color='#E74C3C'
            ))

            fig.update_layout(
                title="Gols Marcados e Sofridos por Time",
                title_x=0.5,
                barmode='group',
                xaxis_title="",
                yaxis_title="Número de Gols",
                plot_bgcolor='white',
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis_tickangle=-45
            )

            fig.update_xaxes(gridcolor='lightgrey')
            fig.update_yaxes(gridcolor='lightgrey')

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Erro ao gerar estatísticas: {str(e)}")
    else:
        st.error("❌ Dados não encontrados")

# Tab Análise de Time
with tab4:
    st.header("🔍 Análise de Time")

    df = load_data()
    if df is not None:
        # Seleção do time
        time_selecionado = st.selectbox(
            "Selecione um Time",
            options=sorted(df['time_casa'].unique())
        )

        try:
            # Filtrar jogos do time
            jogos_time = df[
                (df['time_casa'] == time_selecionado) |
                (df['time_fora'] == time_selecionado)
                ].copy()

            if not jogos_time.empty:
                # Adicionar coluna de resultado
                jogos_time['resultado'] = jogos_time.apply(
                    lambda x: 'Vitória' if (
                            (x['time_casa'] == time_selecionado and x['vencedor'] == 'HOME_TEAM') or
                            (x['time_fora'] == time_selecionado and x['vencedor'] == 'AWAY_TEAM')
                    ) else 'Derrota' if (
                            (x['time_casa'] == time_selecionado and x['vencedor'] == 'AWAY_TEAM') or
                            (x['time_fora'] == time_selecionado and x['vencedor'] == 'HOME_TEAM')
                    ) else 'Empate',
                    axis=1
                )

                # Métricas principais
                st.markdown("### 📊 Desempenho Geral")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    vitorias = len(jogos_time[jogos_time['resultado'] == 'Vitória'])
                    st.markdown(
                        f"""
                        <div class="metric-card" style="text-align: center;">
                            <h4>Vitórias</h4>
                            <h2 style="color: #2ECC71;">{vitorias}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:
                    empates = len(jogos_time[jogos_time['resultado'] == 'Empate'])
                    st.markdown(
                        f"""
                        <div class="metric-card" style="text-align: center;">
                            <h4>Empates</h4>
                            <h2 style="color: #F1C40F;">{empates}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col3:
                    derrotas = len(jogos_time[jogos_time['resultado'] == 'Derrota'])
                    st.markdown(
                        f"""
                        <div class="metric-card" style="text-align: center;">
                            <h4>Derrotas</h4>
                            <h2 style="color: #E74C3C;">{derrotas}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col4:
                    aproveitamento = (vitorias * 3 + empates) / (len(jogos_time) * 3) * 100
                    st.markdown(
                        f"""
                        <div class="metric-card" style="text-align: center;">
                            <h4>Aproveitamento</h4>
                            <h2>{aproveitamento:.1f}%</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Gráficos de análise
                st.markdown("### 📈 Análises Detalhadas")
                col1, col2 = st.columns(2)

                with col1:
                    # Gráfico de pizza com resultados
                    resultados = jogos_time['resultado'].value_counts()

                    fig = go.Figure(data=[go.Pie(
                        labels=resultados.index,
                        values=resultados.values,
                        hole=.3,
                        marker_colors=['#2ECC71', '#F1C40F', '#E74C3C']
                    )])

                    fig.update_layout(
                        title=f"Resultados do {time_selecionado}",
                        title_x=0.5,
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Gráfico de desempenho casa vs fora
                    jogos_casa = jogos_time[jogos_time['time_casa'] == time_selecionado]
                    jogos_fora = jogos_time[jogos_time['time_fora'] == time_selecionado]

                    vit_casa = len(jogos_casa[jogos_casa['vencedor'] == 'HOME_TEAM'])
                    vit_fora = len(jogos_fora[jogos_fora['vencedor'] == 'AWAY_TEAM'])

                    emp_casa = len(jogos_casa[jogos_casa['vencedor'] == 'DRAW'])
                    emp_fora = len(jogos_fora[jogos_fora['vencedor'] == 'DRAW'])

                    der_casa = len(jogos_casa[jogos_casa['vencedor'] == 'AWAY_TEAM'])
                    der_fora = len(jogos_fora[jogos_fora['vencedor'] == 'HOME_TEAM'])

                    fig = go.Figure(data=[
                        go.Bar(name='Vitórias', x=['Casa', 'Fora'], y=[vit_casa, vit_fora], marker_color='#2ECC71'),
                        go.Bar(name='Empates', x=['Casa', 'Fora'], y=[emp_casa, emp_fora], marker_color='#F1C40F'),
                        go.Bar(name='Derrotas', x=['Casa', 'Fora'], y=[der_casa, der_fora], marker_color='#E74C3C')
                    ])

                    fig.update_layout(
                        title=f"Desempenho Casa vs Fora",
                        title_x=0.5,
                        barmode='group',
                        height=400,
                        plot_bgcolor='white'
                    )

                    fig.update_xaxes(gridcolor='lightgrey')
                    fig.update_yaxes(gridcolor='lightgrey')

                    st.plotly_chart(fig, use_container_width=True)

                # Últimos jogos
                st.markdown("### ⚽ Últimos Jogos")
                ultimos_jogos = jogos_time.sort_values('data', ascending=False).head(5)

                for _, jogo in ultimos_jogos.iterrows():
                    data = jogo['data'].strftime('%d/%m/%Y')
                    if jogo['time_casa'] == time_selecionado:
                        resultado = "✅" if jogo['vencedor'] == 'HOME_TEAM' else \
                            "❌" if jogo['vencedor'] == 'AWAY_TEAM' else "➖"
                        cor_fundo = "#d4edda" if jogo['vencedor'] == 'HOME_TEAM' else \
                            "#f8d7da" if jogo['vencedor'] == 'AWAY_TEAM' else "#fff3cd"
                    else:
                        resultado = "✅" if jogo['vencedor'] == 'AWAY_TEAM' else \
                            "❌" if jogo['vencedor'] == 'HOME_TEAM' else "➖"
                        cor_fundo = "#d4edda" if jogo['vencedor'] == 'AWAY_TEAM' else \
                            "#f8d7da" if jogo['vencedor'] == 'HOME_TEAM' else "#fff3cd"

                    st.markdown(
                        f"""
                                        <div style="padding: 10px; background-color: {cor_fundo}; 
                                                border-radius: 5px; margin: 5px 0;">
                                            {resultado} {data} - {jogo['time_casa']} {jogo['gols_casa']} x 
                                            {jogo['gols_fora']} {jogo['time_fora']}
                                        </div>
                                        """,
                        unsafe_allow_html=True
                    )

                # Tendências de gols
                st.markdown("### 📈 Tendências de Gols")

                # Preparar dados para o gráfico
                jogos_time['gols_marcados'] = jogos_time.apply(
                    lambda x: x['gols_casa'] if x['time_casa'] == time_selecionado
                    else x['gols_fora'],
                    axis=1
                )
                jogos_time['gols_sofridos'] = jogos_time.apply(
                    lambda x: x['gols_fora'] if x['time_casa'] == time_selecionado
                    else x['gols_casa'],
                    axis=1
                )

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=jogos_time['data'],
                    y=jogos_time['gols_marcados'],
                    name='Gols Marcados',
                    line=dict(color='#2ECC71', width=3),
                    mode='lines+markers'
                ))

                fig.add_trace(go.Scatter(
                    x=jogos_time['data'],
                    y=jogos_time['gols_sofridos'],
                    name='Gols Sofridos',
                    line=dict(color='#E74C3C', width=3),
                    mode='lines+markers'
                ))

                fig.update_layout(
                    title=f"Tendência de Gols - {time_selecionado}",
                    title_x=0.5,
                    xaxis_title="Data",
                    yaxis_title="Gols",
                    plot_bgcolor='white',
                    height=400,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                fig.update_xaxes(gridcolor='lightgrey')
                fig.update_yaxes(gridcolor='lightgrey')

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("⚠️ Nenhum jogo encontrado para este time")

        except Exception as e:
            st.error(f"❌ Erro na análise de desempenho: {str(e)}")
    else:
        st.error("❌ Dados não encontrados")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666666;'>
            <p>Desenvolvido para análise do Campeonato Brasileiro Série A 2024</p>
            <p>Dados fornecidos por football-data.org</p>
        </div>
        """,
        unsafe_allow_html=True
    )