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

import pulp
solver = pulp.PULP_CBC_CMD(msg=False)

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
    # command = f'cbc single_period_{budget}.mps solve solu solution_sp_{budget}.txt'
    cbc_path = pulp.apis.PULP_CBC_CMD().path
    command = f"{cbc_path} single_period_{budget}.mps solve solu solution_sp_{budget}.txt"
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

    # picks_df = pd.DataFrame(picks, columns=['name', 'pos', 'team','price', 'xP','lineup', 'DC_flag', 'type']).sort_values(by=['lineup', 'type'], ascending=[False, True])
    picks_df = pd.DataFrame(picks, columns=['name', 'pos', 'type', 'team', 'photo','price', 'xP','lineup', 'captain', 'vicecaptain', 'DC_flag']).sort_values(by=['lineup', 'type'], ascending=[False, True])
    # total_xp = so.expr_sum((lineup[p] + captain[p]) * data.loc[p, f'{next_gw}_Pts'] for p in players).get_value()

    # print(f'Total expected value for budget {budget}: {total_xp}')

    return picks_df

def plot_team(picks_df):
    picks_df_plot = picks_df.copy()
    picks_df_plot_lineup = picks_df_plot[picks_df_plot['lineup'] == 1]
    picks_df_plot_lineup = picks_df_plot_lineup.sort_values(by='type', ascending=False)

    formation_dic = picks_df_plot_lineup['pos'].value_counts().sort_index().to_dict()
    formation = [ formation_dic.get('DEF', 4), formation_dic.get('MID', 4), formation_dic.get('FWD', 2)]
    formation_str = ''.join(map(str, formation))
    print(formation)

    picks_df_plot_lineup['photo'] = 'https://resources.premierleague.com/premierleague/photos/players/110x140/p'+picks_df_plot_lineup['photo'].astype(str).str[:-4]+'.png'
    # picks_df_plot_lineup['position_id'] = [11, 9, 10, 8, 4, 7, 3, 6, 5, 2, 1]
    # picks_df_plot_lineup['position_id'] = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    picks_df_plot_lineup['position_id'] = [1, 6, 5, 4, 2, 3, 7, 11, 8, 10, 9][::-1]

    picks_df_plot_bench = picks_df_plot[picks_df_plot['lineup'] == 0]
    picks_df_plot_bench['photo'] = 'https://resources.premierleague.com/premierleague/photos/players/110x140/p'+picks_df_plot_bench['photo'].astype(str).str[:-4]+'.png'
    picks_df_plot_bench_name = picks_df_plot_bench['name'].to_list()
    picks_df_plot_bench_team = picks_df_plot_bench['team'].to_list()
    picks_df_plot_bench_price = picks_df_plot_bench['price'].to_list()
    picks_df_plot_bench_xP = picks_df_plot_bench['xP'].round(1).to_list()
    picks_df_plot_bench_photo = picks_df_plot_bench['photo'].to_list()
    picks_df_plot_lineup['xP'] = picks_df_plot_lineup['xP'].round(1)

    totw_player_data = picks_df_plot_lineup

    images = [Image.open(urlopen(url)) for url in totw_player_data['photo']]

    totw_player_data['player'] = totw_player_data['name']
    totw_player_data['score'] = totw_player_data['xP']
    totw_player_data = totw_player_data.reset_index()
    
    
    formation_dic = picks_df_plot_lineup['pos'].value_counts().sort_index().to_dict()
    formation = [ formation_dic.get('DEF', 4), formation_dic.get('MID', 4), formation_dic.get('FWD', 2)]
    formation_str = ''.join(map(str, formation))
    print(formation_str)

    # setup figure
    pitch = VerticalPitch(pitch_type='opta', pitch_color='teal', line_color='white', line_alpha=0.2,
                        line_zorder=3)
    fig, axes = pitch.grid(endnote_height=0.17, figheight=8, title_height=0.05, title_space=0, space=0)
    fig.set_facecolor('teal')

    # title
    axes['title'].axis('off')
    # axes['title'].text(0.5, 0.9, 'GW 10', ha='center', va='bottom', color='white',fontsize=12)
    # axes['title'].text(0.25, 0.8, 'Sell', ha='right', va='center', color='white',fontsize=10)
    # axes['title'].text(0.75, 0.8, 'Buy', ha='right', va='center', color='white',fontsize=10)
    # Buy: M.Salah (pr:127 xp:6.26 ),Strand Larsen (pr:55 xp:4.21 ),Cunha (pr:66 xp:4.33 ),
    # Sell: Luis Díaz (pr:78 xp:3.88 ),N.Jackson (pr:79 xp:2.96 ),Watkins (pr:91 xp:3.29 ),
    # axes['title'].text(0.5, 0.7, 'Buy: M.Salah (pr:127 xp:6.26 ),Strand Larsen (pr:55 xp:4.21 )', ha='center', va='center', color='black',fontsize=9, bbox=dict(facecolor='greenyellow', boxstyle='round,pad=0.2', linewidth=0))
    # axes['title'].text(0.5, 0.5, 'Sell: Luis Díaz (pr:78 xp:3.88 ),N.Jackson (pr:79 xp:2.96 )', ha='center', va='center', color='black',fontsize=9, bbox=dict(facecolor='tomato', boxstyle='round,pad=0.2', linewidth=0))
    # axes['title'].text(0.5, 0.5, '⇔', ha='center', va='center', color='white',fontsize=16)

    # title_image = inset_image((0.75+1)/2, 0.6, image, height=0.3, ax=axes['title'])

    # axes['title'].text(0.6, 0.7, ' M.Salah (pr:127 xp:6.26 ),Strand Larsen (pr:55 xp:4.21 )', ha='center', va='center', color='white',fontsize=10)

    axes['title'].text(0.5, 1, 'Gameweek 1 team', ha='center', va='center', color='white',
                    fontsize=11)

    axes['title'].text(0.5, 0.05, f"Total Cost: {picks_df['price'].sum()}", ha='center', va='center', color='white',
                    fontsize=9, bbox=dict(facecolor='darkred', boxstyle='round,pad=0.2', linewidth=0))

    # axes['title'].text(0.5, 0.3, 'Round 9', ha='center', va='center', color='white', fontsize=14)

    # plot the league logo using the inset_image method for utils
    # LEAGUE_URL = 'https://www.thesportsdb.com/images/media/league/badge/kxo7zf1656519439.png'
    # image = Image.open(urlopen(LEAGUE_URL))
    # title_image = inset_image(0.9, 0.5, image, height=1, ax=axes['title'])

    axes['endnote'].axis('off')
    LEAGUE_URL = 'https://resources.premierleague.com/premierleague/photos/players/110x140/p219847.png'
    image = Image.open(urlopen(LEAGUE_URL))
    # Reduce the size of the image
    # new_size = (20, 20)  # New size (width, height)
    # image = image.resize(new_size, Image.LANCZOS)
    # title_image = inset_image((0+0.25)/2, 0.8, Image.open(urlopen(picks_df_plot_bench_photo[0])), height=0.5, ax=axes['endnote'])
    # title_image = inset_image((0.25+0.5)/2, 0.8, Image.open(urlopen(picks_df_plot_bench_photo[1])), height=0.5, ax=axes['endnote'])
    # title_image = inset_image((0.5+0.75)/2, 0.8, Image.open(urlopen(picks_df_plot_bench_photo[2])), height=0.5, ax=axes['endnote'])
    # title_image = inset_image((0.75+1)/2, 0.8, Image.open(urlopen(picks_df_plot_bench_photo[3])), height=0.5, ax=axes['endnote'])
    try:
        title_image = inset_image((0+0.25)/2, 0.8, Image.open(urlopen(picks_df_plot_bench_photo[0])), height=0.5, ax=axes['endnote'])
    except Exception as e:
        print(f"Bench photo 0 not loaded: {e}")

    try:
        title_image = inset_image((0.25+0.5)/2, 0.8, Image.open(urlopen(picks_df_plot_bench_photo[1])), height=0.5, ax=axes['endnote'])
    except Exception as e:
        print(f"Bench photo 1 not loaded: {e}")

    try:
        title_image = inset_image((0.5+0.75)/2, 0.8, Image.open(urlopen(picks_df_plot_bench_photo[2])), height=0.5, ax=axes['endnote'])
    except Exception as e:
        print(f"Bench photo 2 not loaded: {e}")

    try:
        title_image = inset_image((0.75+1)/2, 0.8, Image.open(urlopen(picks_df_plot_bench_photo[3])), height=0.5, ax=axes['endnote'])
    except Exception as e:
        print(f"Bench photo 3 not loaded: {e}")
    # axes['endnote'].text(0.9, 0.1, 'Havertz', ha='center', va='center', color='white',
    #                    fontsize=11)

    # Add the player name below the image
    # axes['endnote'].text((0+0.25)/2, 0.4, picks_df_plot_bench_name[0], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes)
    # axes['endnote'].text((0.25+0.5)/2, 0.4, picks_df_plot_bench_name[1], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes)
    # axes['endnote'].text((0.5+0.75)/2, 0.4, picks_df_plot_bench_name[2], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes)
    # axes['endnote'].text((0.75+1)/2, 0.4, picks_df_plot_bench_name[3], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes)

    # axes['endnote'].text((0+0.25)/2, 0.25, picks_df_plot_bench_xP[0], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='green', boxstyle='round,pad=0.2', linewidth=0))
    # axes['endnote'].text((0.25+0.5)/2, 0.25, picks_df_plot_bench_xP[1], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='green', boxstyle='round,pad=0.2', linewidth=0))
    # axes['endnote'].text((0.5+0.75)/2, 0.25, picks_df_plot_bench_xP[2], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='green', boxstyle='round,pad=0.2', linewidth=0))
    # axes['endnote'].text((0.75+1)/2, 0.25, picks_df_plot_bench_xP[3], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='green', boxstyle='round,pad=0.2', linewidth=0))

    # axes['endnote'].text((0+0.25)/2, 0.1, picks_df_plot_bench_price[0], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='tomato', boxstyle='round,pad=0.2', linewidth=0))
    # axes['endnote'].text((0.25+0.5)/2, 0.1, picks_df_plot_bench_price[1], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='tomato', boxstyle='round,pad=0.2', linewidth=0))
    # axes['endnote'].text((0.5+0.75)/2, 0.1, picks_df_plot_bench_price[2], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='tomato', boxstyle='round,pad=0.2', linewidth=0))
    # axes['endnote'].text((0.75+1)/2, 0.1, picks_df_plot_bench_price[3], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='tomato', boxstyle='round,pad=0.2', linewidth=0))

    axes['endnote'].text((0+0.25)/2, 0.4, picks_df_plot_bench_name[0], ha='center', va='center', color='black', fontsize=7, transform=axes['endnote'].transAxes, bbox=dict(facecolor='white', boxstyle='round,pad=0.2', linewidth=0))
    axes['endnote'].text((0.25+0.5)/2, 0.4, picks_df_plot_bench_name[1], ha='center', va='center', color='black', fontsize=7, transform=axes['endnote'].transAxes, bbox=dict(facecolor='white', boxstyle='round,pad=0.2', linewidth=0))
    axes['endnote'].text((0.5+0.75)/2, 0.4, picks_df_plot_bench_name[2], ha='center', va='center', color='black', fontsize=7, transform=axes['endnote'].transAxes, bbox=dict(facecolor='white', boxstyle='round,pad=0.2', linewidth=0))
    axes['endnote'].text((0.75+1)/2, 0.4, picks_df_plot_bench_name[3], ha='center', va='center', color='black', fontsize=7, transform=axes['endnote'].transAxes, bbox=dict(facecolor='white', boxstyle='round,pad=0.2', linewidth=0))

    axes['endnote'].text((0+0.25)/2, 0.25, picks_df_plot_bench_team[0], ha='center', va='center', color='white', fontsize=7, transform=axes['endnote'].transAxes)
    axes['endnote'].text((0.25+0.5)/2, 0.25, picks_df_plot_bench_team[1], ha='center', va='center', color='white', fontsize=7, transform=axes['endnote'].transAxes)
    axes['endnote'].text((0.5+0.75)/2, 0.25, picks_df_plot_bench_team[2], ha='center', va='center', color='white', fontsize=7, transform=axes['endnote'].transAxes)
    axes['endnote'].text((0.75+1)/2, 0.25, picks_df_plot_bench_team[3], ha='center', va='center', color='white', fontsize=7, transform=axes['endnote'].transAxes)

    axes['endnote'].text((0+0.25)/2 - 0.05, 0.1, picks_df_plot_bench_xP[0], ha='center', va='center', color='black', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='palegreen', boxstyle='round,pad=0.2', linewidth=0))
    axes['endnote'].text((0.25+0.5)/2- 0.05, 0.1, picks_df_plot_bench_xP[1], ha='center', va='center', color='black', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='palegreen', boxstyle='round,pad=0.2', linewidth=0))
    axes['endnote'].text((0.5+0.75)/2- 0.05, 0.1, picks_df_plot_bench_xP[2], ha='center', va='center', color='black', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='palegreen', boxstyle='round,pad=0.2', linewidth=0))
    axes['endnote'].text((0.75+1)/2- 0.05, 0.1, picks_df_plot_bench_xP[3], ha='center', va='center', color='black', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='palegreen', boxstyle='round,pad=0.2', linewidth=0))

    axes['endnote'].text((0+0.25)/2+ 0.05, 0.1, picks_df_plot_bench_price[0], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='darkred', boxstyle='round,pad=0.2', linewidth=0))
    axes['endnote'].text((0.25+0.5)/2+ 0.05, 0.1, picks_df_plot_bench_price[1], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='darkred', boxstyle='round,pad=0.2', linewidth=0))
    axes['endnote'].text((0.5+0.75)/2+ 0.05, 0.1, picks_df_plot_bench_price[2], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='darkred', boxstyle='round,pad=0.2', linewidth=0))
    try:
        axes['endnote'].text((0.75+1)/2+ 0.05, 0.1, picks_df_plot_bench_price[3], ha='center', va='center', color='white', fontsize=9, transform=axes['endnote'].transAxes, bbox=dict(facecolor='darkred', boxstyle='round,pad=0.2', linewidth=0))
    except:
        pass



    text_names = pitch.formation(formation_str, kind='text', 
                                positions=totw_player_data.position_id,
                                text=totw_player_data['name'], ax=axes['pitch'], flip = True,
                                xoffset=-2,  # offset the player names from the centers
                                ha='center', va='center', color='black', fontsize=7,
                                bbox=dict(facecolor='white', boxstyle='round,pad=0.2', linewidth=0))

    names = pitch.formation(formation_str, kind='text', 
                                positions=totw_player_data.position_id,
                                text=totw_player_data['team'], ax=axes['pitch'], flip = True,
                                xoffset=-5,  # offset the player names from the centers
                                ha='center', va='center', color='white', fontsize=7)
    text_scores = pitch.formation(formation_str, kind='text', 
                                positions=totw_player_data.position_id,
                                text=totw_player_data['xP'], ax=axes['pitch'],flip = True,
                                xoffset=-9,yoffset=5,  # offset the scores from the centers
                                ha='center', va='center', color='black', fontsize=9,
                                bbox=dict(facecolor='palegreen', boxstyle='round,pad=0.2', linewidth=0))

    text_scores = pitch.formation(formation_str, kind='text', 
                                positions=totw_player_data.position_id,
                                text=totw_player_data['price'], ax=axes['pitch'],flip = True,
                                xoffset=-9,yoffset=-5,  # offset the scores from the centers
                                ha='center', va='center', color='white', fontsize=9,
                                bbox=dict(facecolor='darkred', boxstyle='round,pad=0.2', linewidth=0))
    badge_axes = pitch.formation(formation_str, kind='image', 
                                positions=totw_player_data.position_id,
                                image=images, height=10, ax=axes['pitch'],flip = True,
                                xoffset=5,  # offset the images from the centers
                                )


    # plt.savefig('FPL/442.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    # plt.show()
    return fig



st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Select Section",
    ["FPL Optimization", "Expected points as per last season"]
)

if "picks_df" not in st.session_state:
    st.session_state.picks_df = None

if page == "FPL Optimization":
    st.header("FPL Optimization")
    st.subheader("Try running with no constraints first, then add constraints to see how it affects the picks.")
    st.write("This tool helps you optimize your Fantasy Premier League (FPL) team based on last year's points (see next tab for details) and various constraints.")

    obj_func_map = {
        'First 10 weeks': 'target_point_for_opt_10w',
        'First 5 weeks': 'target_point_for_opt_5w'
    }
    obj_func_input_temp = st.pills("Objective function", ["First 10 weeks", "First 5 weeks"], default = 'First 10 weeks', selection_mode="single")
    obj_func_input = obj_func_map[obj_func_input_temp]


    
    player_name_map = dict(zip(merged_data['web_name'], merged_data['player_name']))
    incl_player = st.multiselect(
        "Include players (Forces the player in your final squad)",
        merged_data['web_name'].to_list(),
        default=[]
    )
    incl_player_input = [player_name_map[item] for item in incl_player]
    excl_player = st.multiselect(
        "Exclude players",
        merged_data['web_name'].to_list(),
        default=[]
    )
    excl_player_input = [player_name_map[item] for item in excl_player]

    excl_teams = st.multiselect(
        "Exclude teams",
        list(merged_data['team'].unique()),
        default=[]
    )

    double_def_input = st.multiselect(
        "Force Double Defence",
        list(merged_data['team'].unique()),
        default=[]
    )

    # DC_input = st.button('Force Defensive Contribution')
    # if DC_input:
    n_DC_input = st.number_input("Number of players with defensive contribution", min_value=0, max_value=5, value=0, step=1)

    DC_imput_final = False if n_DC_input == 0 else n_DC_input


    if st.button("Run Optimization"):
        try:
            picks_df = run_opt(merged_data, obj_func=obj_func_input, include_players=incl_player_input, exclude_players=excl_player_input, exclude_teams=excl_teams, double_def=double_def_input, n_DC=DC_imput_final)
            print(picks_df['price'].sum())
    
            # st.dataframe(picks_df)
            st.session_state.picks_df = picks_df
        except:
            st.title('ERROR: Check what you have selected')

    if st.session_state.picks_df is not None:
        
        st.pyplot(plot_team(st.session_state.picks_df))
        st.write("Selected Players:")
        st.dataframe(st.session_state.picks_df[['name', 'pos', 'team','price', 'xP','lineup', 'DC_flag']])

elif page == "Expected points as per last season":
    st.title("Expected Points as per Last Season")
    
    # st.write("This section is under development. Please check back later for updates.")
    st.image('fwd.png')
    st.image('mid.png')
    st.image('def.png')
    st.image('gk.png')
    # Placeholder for future content
    # You can add charts, tables, or any other relevant information here.

















