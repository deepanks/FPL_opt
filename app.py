import streamlit as st

import pandas as pd
import numpy as np
import sasoptpy as so
import requests
import os
import time

import math
from urllib.request import urlopen

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import subprocess

from mplsoccer import VerticalPitch, Sbopen, FontManager, inset_image

budget = 100
# budget = 98.7
# budget = 98.5

r = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
fpl_data = r.json()
element_data = pd.DataFrame(fpl_data['elements'])
team_data = pd.DataFrame(fpl_data['teams'])
elements_team = pd.merge(element_data, team_data, left_on='team', right_on='id')
elements_team['name_fn_ln'] = elements_team['first_name'] + ' ' + elements_team['second_name']
elements_team.head()

df_points = pd.read_csv('fpl_player_points_sim_2024_25_v2_csv.csv')

## adjustments
# man_u_adj = {
#     'Bruno Borges Fernandes',
#     'Bryan Mbeumo',
#     'Matheus Santos Carneiro da Cunha'
# }



for each in ['id_x','element_type', 'now_cost' , 'web_name', 'name', 'photo']:
    df_points[each] = df_points['player_name'].map(dict(zip(elements_team['name_fn_ln'], elements_team[each])))
    print(each,' not available for:', df_points[each].isnull().sum())

merged_data = df_points.copy()
merged_data.set_index(['id_x'], inplace=True)

merged_data['target_point_for_opt_10w'] = merged_data['first_10_mean'].fillna(0)
merged_data['target_point_for_opt_5w'] = merged_data['first_5_mean'].fillna(0)

def_contri_pt = {
    "Joško Gvardiol": 10,
    "Virgil van Dijk": 22,
    "Nikola Milenković": 26,
    "Daniel Muñoz Mejía": 14,
    "Murillo Costa dos Santos": 38,
    "James Tarkowski": 44,
    "Jarrad Branthwaite": 32,
    "Nathan Collins": 32,
    "Maxence Lacroix": 34
}

merged_data['def_contri_flag'] = np.where(merged_data['player_name'].isin(def_contri_pt.keys()), 1, 0)

# merged_data['target_point_for_opt_10w'] = np.where(merged_data['player_name'].isin(man_u_adj), merged_data['target_point_for_opt_10w']*0.75, merged_data['target_point_for_opt_10w'])
# merged_data['target_point_for_opt_5w'] = np.where(merged_data['player_name'].isin(man_u_adj), merged_data['target_point_for_opt_5w']*0.75, merged_data['target_point_for_opt_10w'])

print(merged_data.shape)
merged_data.head()


def run_opt(data ,obj_func = '', include_players = [], exclude_players = [], exclude_teams = [], double_def=[], n_DC = False):
    type_data = pd.DataFrame(fpl_data['element_types']).set_index(['id'])
    

    players = data.index.to_list()
    element_types = type_data.index.to_list()
    teams = team_data['name'].to_list()
    model = so.Model(name='single_period')

    squad = model.add_variables(players, name='squad', vartype=so.binary)
    lineup = model.add_variables(players, name='lineup', vartype=so.binary)
    captain = model.add_variables(players, name='captain', vartype=so.binary)
    vicecap = model.add_variables(players, name='vicecap', vartype=so.binary)

    # Constraints
    squad_count = so.expr_sum(squad[p] for p in players)
    model.add_constraint(squad_count == 15, name='squad_count')
    model.add_constraint(so.expr_sum(lineup[p] for p in players) == 11, name='lineup_count')
    model.add_constraint(so.expr_sum(captain[p] for p in players) == 1, name='captain_count')
    model.add_constraint(so.expr_sum(vicecap[p] for p in players) == 1, name='vicecap_count')
    model.add_constraints((lineup[p] <= squad[p] for p in players), name='lineup_squad_rel')
    model.add_constraints((captain[p] <= lineup[p] for p in players), name='captain_lineup_rel')
    model.add_constraints((vicecap[p] <= lineup[p] for p in players), name='vicecap_lineup_rel')
    model.add_constraints((captain[p] + vicecap[p] <= 1 for p in players), name='cap_vc_rel')
    lineup_type_count = {t: so.expr_sum(lineup[p] for p in players if data.loc[p, 'element_type'] == t) for t in element_types}
    squad_type_count = {t: so.expr_sum(squad[p] for p in players if data.loc[p, 'element_type'] == t) for t in element_types}
    model.add_constraints((lineup_type_count[t] == [type_data.loc[t, 'squad_min_play'], type_data.loc[t, 'squad_max_play']] for t in element_types), name='valid_formation')
    model.add_constraints((squad_type_count[t] == type_data.loc[t, 'squad_select'] for t in element_types), name='valid_squad')
    price = so.expr_sum(data.loc[p, 'now_cost'] / 10 * squad[p] for p in players)
    model.add_constraint(price <= budget, name='budget_limit')
    model.add_constraints((so.expr_sum(squad[p] for p in players if data.loc[p, 'name'] == t) <= 3 for t in teams), name='team_limit')

    
    model.add_constraints(so.expr_sum(squad[merged_data[merged_data['player_name'] == p].index.to_list()[0]] for p in include_players) == len(include_players), name = 'inc_players')
    model.add_constraints(so.expr_sum(squad[merged_data[merged_data['player_name'] == p].index.to_list()[0]] for p in exclude_players) == 0, name = 'exc_players')
    # model.add_constraints((so.expr_sum(squad[p] for p in players if (data.loc[p, 'name'] == t)) == 0 for t in exclude_teams), name='exclude_teams')
    model.add_constraints((so.expr_sum(squad[p] for p in players if (data.loc[p, 'name'] == t and p not in merged_data[merged_data['player_name'].isin(include_players)].index)) == 0 for t in exclude_teams), name='exclude_teams')
    

    model.add_constraints((so.expr_sum(lineup[p] for p in players if (data.loc[p, 'name'] == t and data.loc[p, 'element_type'] == 2)) == 2 for t in double_def), name='double_defence')

    if n_DC:
        model.add_constraints( so.expr_sum(squad[p] for p in players if data.loc[p, 'def_contri_flag'] == 1) == n_DC, name='def_contri_flag')

    # model.add_constraints(so.expr_sum(squad[p] ))
    # model.add_constraints(squad[merged_data[merged_data['player_name'] == 'Erling Haaland'].index.to_list()[0]] == 1, name = 'noHalland')

    # total_points = so.expr_sum(data.loc[p, f'{next_gw}_Pts'] * (lineup[p] + captain[p] + 0.1 * vicecap[p]) for p in players)
    # total_points = so.expr_sum(data.loc[p, 'target_point_for_opt'] * (lineup[p] + captain[p] + 0.1 * vicecap[p]) for p in players)
    total_points = so.expr_sum(data.loc[p, obj_func] * (lineup[p] + 0.1*captain[p] + 0.1 * vicecap[p] + 0.01*squad[p]) for p in players)

    model.set_objective(-total_points, sense='N', name='total_xp')

    model.export_mps(f'single_period_{budget}.mps')
    # command = f'"C:/Program Files/CBC Solver/bin/cbc.exe" {problem_name}.mps solve solu {problem_name}_sol.txt'
    command = f'"cbc.exe" single_period_{budget}.mps solve solu solution_sp_{budget}.txt'
    # !{command}
    subprocess.run(command, shell=True, check=True)

    for v in model.get_variables():
        v.set_value(0)
    with open(f'solution_sp_{budget}.txt', 'r') as f:
        for line in f:
            if 'objective value' in line:
                continue
            words = line.split()
            var = model.get_variable(words[1])
            var.set_value(float(words[2]))

    picks = []
    for p in players:
        if squad[p].get_value() > 0.5:
            print(f'Player {p} is selected in the squad')
            lp = data.loc[p]
            is_captain = 1 if captain[p].get_value() > 0.5 else 0
            is_lineup = 1 if lineup[p].get_value() > 0.5 else 0
            is_vice = 1 if vicecap[p].get_value() > 0.5 else 0
            position = type_data.loc[lp['element_type'], 'singular_name_short']
            picks.append([
                lp['web_name'], position, lp['element_type'], lp['name'],lp['photo'], lp['now_cost']/10, lp[obj_func] ,is_lineup, is_captain, is_vice, lp['def_contri_flag']
            ])

    picks_df = pd.DataFrame(picks, columns=['name', 'pos', 'type', 'team', 'photo','price', 'xP','lineup', 'captain', 'vicecaptain', 'DC_flag']).sort_values(by=['lineup', 'type'], ascending=[False, True])
    # total_xp = so.expr_sum((lineup[p] + captain[p]) * data.loc[p, f'{next_gw}_Pts'] for p in players).get_value()

    # print(f'Total expected value for budget {budget}: {total_xp}')

    return picks_df


incl_player = st.multiselect(
    "Include players",
    merged_data['player_name'].to_list(),
    default=[],
)




picks_df = run_opt(merged_data, obj_func='target_point_for_opt_10w', include_players=incl_player, exclude_players=[], exclude_teams=[ ], double_def=['Arsenal'], n_DC=False)
print(picks_df['price'].sum())


st.dataframe(picks_df)


