import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import os


class BrasileiraoDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.features = None

        logging.basicConfig(
            filename='logs/data_processing.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def get_team_position(self, time):
        """Obtém a posição atual do time na tabela"""
        try:
            classificacao = pd.read_csv('data/classificacao.csv')
            time_info = classificacao[classificacao['time'] == time]
            if not time_info.empty:
                return time_info.iloc[0]['posicao']
            return None
        except:
            return None

    def calcular_estatisticas_time(self, df, nome_time, ultimas_n_partidas=5):
        """Calcula estatísticas recentes de um time"""
        # Jogos em casa e fora
        jogos_casa = df[df['time_casa'] == nome_time].copy()
        jogos_fora = df[df['time_fora'] == nome_time].copy()

        # Últimos jogos com peso maior para jogos mais recentes
        todos_jogos = pd.concat([jogos_casa, jogos_fora]).sort_values('data', ascending=False)
        ultimos_jogos = todos_jogos.head(ultimas_n_partidas)

        if len(ultimos_jogos) < 3:
            return None

        # Calcular forma recente com pesos
        pesos = np.array([1.0, 0.8, 0.6, 0.4, 0.2])[:len(ultimos_jogos)]
        pontos_recentes = []

        for _, jogo in ultimos_jogos.iterrows():
            if jogo['time_casa'] == nome_time:
                if jogo['vencedor'] == 'HOME_TEAM':
                    pontos_recentes.append(3)
                elif jogo['vencedor'] == 'DRAW':
                    pontos_recentes.append(1)
                else:
                    pontos_recentes.append(0)
            else:
                if jogo['vencedor'] == 'AWAY_TEAM':
                    pontos_recentes.append(3)
                elif jogo['vencedor'] == 'DRAW':
                    pontos_recentes.append(1)
                else:
                    pontos_recentes.append(0)

        forma_ponderada = np.average(pontos_recentes, weights=pesos) if pontos_recentes else 0

        # Obter posição na tabela
        posicao_atual = self.get_team_position(nome_time)
        if posicao_atual is None:
            posicao_atual = 10  # valor médio caso não encontre

        stats = {
            'posicao': posicao_atual,
            'media_gols_pro': float((
                                        jogos_casa['gols_casa'].mean() if not jogos_casa.empty else 0 +
                                                                                                    jogos_fora[
                                                                                                        'gols_fora'].mean() if not jogos_fora.empty else 0
                                    ) / 2),
            'media_gols_contra': float((
                                           jogos_casa['gols_fora'].mean() if not jogos_casa.empty else 0 +
                                                                                                       jogos_fora[
                                                                                                           'gols_casa'].mean() if not jogos_fora.empty else 0
                                       ) / 2),
            'forma_recente': float(forma_ponderada),
            'vitorias_casa': int((jogos_casa['vencedor'] == 'HOME_TEAM').sum()),
            'vitorias_fora': int((jogos_fora['vencedor'] == 'AWAY_TEAM').sum()),
            'derrotas_casa': int((jogos_casa['vencedor'] == 'AWAY_TEAM').sum()),
            'derrotas_fora': int((jogos_fora['vencedor'] == 'HOME_TEAM').sum()),
            'jogos_casa': len(jogos_casa),
            'jogos_fora': len(jogos_fora)
        }

        # Calcular aproveitamentos
        total_pontos_casa = (stats['vitorias_casa'] * 3 +
                             jogos_casa[jogos_casa['vencedor'] == 'DRAW'].shape[0])
        total_pontos_fora = (stats['vitorias_fora'] * 3 +
                             jogos_fora[jogos_fora['vencedor'] == 'DRAW'].shape[0])

        stats['aproveitamento_casa'] = (
            float(total_pontos_casa / (stats['jogos_casa'] * 3))
            if stats['jogos_casa'] > 0 else 0
        )

        stats['aproveitamento_fora'] = (
            float(total_pontos_fora / (stats['jogos_fora'] * 3))
            if stats['jogos_fora'] > 0 else 0
        )

        return stats

    def preparar_features_partida(self, df, time_casa, time_fora):
        """Prepara features para uma partida específica"""
        stats_casa = self.calcular_estatisticas_time(df, time_casa)
        stats_fora = self.calcular_estatisticas_time(df, time_fora)

        if not stats_casa or not stats_fora:
            return None

        features = [
            # Features do time da casa
            float(stats_casa['posicao']),
            float(stats_casa['media_gols_pro']),
            float(stats_casa['media_gols_contra']),
            float(stats_casa['forma_recente']),
            float(stats_casa['aproveitamento_casa']),

            # Features do time visitante
            float(stats_fora['posicao']),
            float(stats_fora['media_gols_pro']),
            float(stats_fora['media_gols_contra']),
            float(stats_fora['forma_recente']),
            float(stats_fora['aproveitamento_fora']),

            # Diferença de posição
            float(stats_fora['posicao'] - stats_casa['posicao'])
        ]

        return features

    def preparar_dados_treino(self, df):
        """Prepara dados para treinamento"""
        features_list = []
        targets = []

        # Usar apenas jogos finalizados
        df = df[df['status'] == 'FINISHED'].sort_values('data')

        for idx, partida in df.iterrows():
            dados_anteriores = df[df['data'] < partida['data']]

            features = self.preparar_features_partida(
                dados_anteriores,
                partida['time_casa'],
                partida['time_fora']
            )

            if features:
                features_list.append(features)
                target = (
                    2 if partida['vencedor'] == 'HOME_TEAM'
                    else 1 if partida['vencedor'] == 'DRAW'
                    else 0
                )
                targets.append(target)

        if not features_list:
            return None, None

        X = np.array(features_list, dtype=np.float64)
        y = np.array(targets, dtype=np.int32)

        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)

        self.features = [
            'posicao_casa',
            'media_gols_pro_casa',
            'media_gols_contra_casa',
            'forma_recente_casa',
            'aproveitamento_casa',
            'posicao_fora',
            'media_gols_pro_fora',
            'media_gols_contra_fora',
            'forma_recente_fora',
            'aproveitamento_fora',
            'diferenca_posicao'
        ]

        return X_scaled, y

    def preparar_dados_predicao(self, df, time_casa, time_fora):
        """Prepara dados para previsão"""
        features = self.preparar_features_partida(df, time_casa, time_fora)

        if not features:
            return None

        return self.scaler.transform(np.array(features).reshape(1, -1))

    def obter_forma_recente(self, df, time, n_jogos=5):
        """Obtém sequência de resultados recentes"""
        jogos = df[
            (df['time_casa'] == time) |
            (df['time_fora'] == time)
            ].sort_values('data', ascending=False).head(n_jogos)

        forma = []
        for _, jogo in jogos.iterrows():
            if jogo['time_casa'] == time:
                if jogo['vencedor'] == 'HOME_TEAM':
                    forma.append('✅')
                elif jogo['vencedor'] == 'AWAY_TEAM':
                    forma.append('❌')
                else:
                    forma.append('➖')
            else:
                if jogo['vencedor'] == 'AWAY_TEAM':
                    forma.append('✅')
                elif jogo['vencedor'] == 'HOME_TEAM':
                    forma.append('❌')
                else:
                    forma.append('➖')

        return forma