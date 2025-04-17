import pandas as pd
import numpy as np
from scipy.stats import poisson

class PoissonPredictor:
    def __init__(self, data):
        self.teams = np.unique(data['HomeTeam'].tolist() + data['AwayTeam'].tolist())
        self.data = data
        self.attack_strength = {}
        self.defense_strength = {}
        self.league_avg_home_goals = data['FTHG'].mean()
        self.league_avg_away_goals = data['FTAG'].mean()
        self._train()

    def _train(self):
        team_stats = {team: {'home_goals': 0, 'away_goals': 0, 'home_games': 0, 'away_games': 0} for team in self.teams}
        for _, row in self.data.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']
            team_stats[home]['home_goals'] += row['FTHG']
            team_stats[home]['home_games'] += 1
            team_stats[away]['away_goals'] += row['FTAG']
            team_stats[away]['away_games'] += 1

        for team in self.teams:
            attack = (team_stats[team]['home_goals'] / team_stats[team]['home_games']) / self.league_avg_home_goals
            defense = (team_stats[team]['away_goals'] / team_stats[team]['away_games']) / self.league_avg_away_goals
            self.attack_strength[team] = attack
            self.defense_strength[team] = defense

    def predict_score(self, home, away):
        home_goals_avg = self.league_avg_home_goals * self.attack_strength[home] * self.defense_strength[away]
        away_goals_avg = self.league_avg_away_goals * self.attack_strength[away] * self.defense_strength[home]
        return round(home_goals_avg, 2), round(away_goals_avg, 2)

    def predict_score_distribution(self, home, away, max_goals=5):
        home_avg, away_avg = self.predict_score(home, away)
        score_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = poisson.pmf(i, home_avg) * poisson.pmf(j, away_avg)
                score_matrix[i][j] = p
        return score_matrix