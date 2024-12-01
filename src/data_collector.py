import requests
import pandas as pd
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import time


class BrasileiraoDataCollector:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('FOOTBALL_API_KEY')
        self.base_url = 'http://api.football-data.org/v4'
        self.headers = {'X-Auth-Token': self.api_key}
        self.competition_id = 2013  # ID do Brasileirão

        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            filename='logs/data_collection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def get_matches(self):
        """Obtém partidas do Brasileirão da temporada atual"""
        try:
            url = f"{self.base_url}/competitions/2013/matches"
            params = {'season': 2023}  # Temporada atual
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            logging.info(f"Dados obtidos com sucesso")
            return response.json()
        except Exception as e:
            logging.error(f"Erro ao obter dados: {str(e)}")
            return None

    def get_team_standing(self):
        """Obtém classificação atual do Brasileirão"""
        try:
            url = f"{self.base_url}/competitions/2013/standings"
            params = {'season': 2023}  # Temporada atual
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Erro ao obter classificação: {str(e)}")
            return None

    def process_matches_data(self, matches_data):
        """Processa dados das partidas"""
        matches_list = []

        for match in matches_data.get('matches', []):
            match_dict = {
                'rodada': match.get('matchday', 0),
                'data': match.get('utcDate', ''),
                'status': match.get('status', ''),
                'time_casa': match.get('homeTeam', {}).get('name', ''),
                'time_fora': match.get('awayTeam', {}).get('name', ''),
                'gols_casa': match.get('score', {}).get('fullTime', {}).get('home', 0),
                'gols_fora': match.get('score', {}).get('fullTime', {}).get('away', 0),
                'vencedor': match.get('score', {}).get('winner', ''),
                'temporada': 2023  # Temporada atual
            }
            matches_list.append(match_dict)

        df = pd.DataFrame(matches_list)
        df['data'] = pd.to_datetime(df['data'])

        # Ordenar por rodada
        df = df.sort_values(['rodada', 'data'])

        logging.info(f"Processados {len(matches_list)} jogos da temporada 2023")
        return df

    def process_standings_data(self, standings_data):
        """Processa dados da classificação"""
        standings_list = []
        try:
            for team in standings_data['standings'][0]['table']:
                team_dict = {
                    'posicao': team.get('position', 0),
                    'time': team.get('team', {}).get('name', ''),
                    'pontos': team.get('points', 0),
                    'jogos': team.get('playedGames', 0),
                    'vitorias': team.get('won', 0),
                    'empates': team.get('draw', 0),
                    'derrotas': team.get('lost', 0),
                    'gols_pro': team.get('goalsFor', 0),
                    'gols_contra': team.get('goalsAgainst', 0),
                    'saldo_gols': team.get('goalDifference', 0)
                }
                standings_list.append(team_dict)

            return pd.DataFrame(standings_list)
        except Exception as e:
            logging.error(f"Erro ao processar classificação: {str(e)}")
            return None

    def update_data(self):
        """Atualiza dados do Brasileirão"""
        try:
            # Coletar dados das partidas
            matches_data = self.get_matches()
            if matches_data is None:
                return None

            # Processar dados das partidas
            df = self.process_matches_data(matches_data)
            if df is None:
                return None

            # Salvar dados
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/brasileirao_matches.csv', index=False)
            logging.info(f"Dados de jogos salvos: {len(df)} partidas")

            # Atualizar classificação
            standings = self.get_team_standing()
            if standings:
                standings_df = self.process_standings_data(standings)
                if standings_df is not None:
                    standings_df.to_csv('data/classificacao.csv', index=False)
                    logging.info("Classificação atualizada")

            return df

        except Exception as e:
            logging.error(f"Erro ao atualizar dados: {str(e)}")
            return None