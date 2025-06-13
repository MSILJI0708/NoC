import pandas as pd
import numpy as np
import time
from numba import njit
from joblib import Parallel, delayed

# 1. 데이터 로딩 및 전처리
url = 'https://www.koreabaseball.com/Record/TeamRank/TeamRankDaily.aspx'
kbo_tb = pd.read_html(url)[0]  # 순위
kbo_tie = pd.read_html(url)[1]  # 상대전적

kbo_tb['팀명'] = kbo_tb['팀명'].str.strip()
kbo_tie['팀명'] = kbo_tie['팀명'].str.strip()
kbo_tie.columns = [col.split(' ')[0] if ' (' in str(col) else col for col in kbo_tie.columns]

teams = kbo_tb['팀명'].tolist()
n_teams = len(teams)
name_to_idx = {name: idx for idx, name in enumerate(teams)}

wins0 = kbo_tb['승'].astype(np.int32).values
losses0 = kbo_tb['패'].astype(np.int32).values
draws0 = kbo_tb['무'].astype(np.int32).values

h2h_wins0 = np.zeros((n_teams, n_teams), dtype=np.int32)
h2h_losses0 = np.zeros((n_teams, n_teams), dtype=np.int32)
h2h_draws0 = np.zeros((n_teams, n_teams), dtype=np.int32)







def clean_tie_table(tie_df, teams):
    for team in teams:
        for opponent in teams:
            if team == opponent:
                tie_df.loc[tie_df['팀명'] == team, opponent] = '0-0-0'
                continue
            try:
                val = tie_df.loc[tie_df['팀명'] == team, opponent].values[0]
                # 값이 문자열이 아니거나, 제대로 된 형식이 아니면 예외 발생
                w, l, d = map(int, str(val).strip().split('-'))
            except Exception:
                tie_df.loc[tie_df['팀명'] == team, opponent] = '0-0-0'
    return tie_df


# 데이터 정제 적용
kbo_tie = clean_tie_table(kbo_tie, teams)

for i, row in kbo_tie.iterrows():
    idx_i = name_to_idx[row['팀명']]
    for team_j, val in row.items():
        if team_j == '팀명' or team_j == '합계': continue
        idx_j = name_to_idx[team_j]
        w, l, d = map(int, str(val).split('-'))
        h2h_wins0[idx_i, idx_j] = w
        h2h_losses0[idx_i, idx_j] = l
        h2h_draws0[idx_i, idx_j] = d

#
def convert_match_teamnames_to_indices(team_match_days, name_to_idx):
    """ 팀명 리스트를 index로 변환 """
    all_matches = []
    for day in team_match_days:
        matches = []
        for match in day:
            t1, t2 = match
            matches.append([name_to_idx[t1], name_to_idx[t2]])
        all_matches.append(matches)
    return all_matches


# 2. Numba 최적화 시뮬레이션 함수
@njit
def simulate_rank_numba(wins0, losses0, draws0,
                         h2h_wins0, h2h_losses0, h2h_draws0,
                         match_list, outcomes):
    n_teams = wins0.shape[0]
    total_games = match_list.shape[0]

    wins = wins0.copy()
    losses = losses0.copy()
    draws = draws0.copy()
    h2h_wins = h2h_wins0.copy()
    h2h_losses = h2h_losses0.copy()
    h2h_draws = h2h_draws0.copy()

    for g in range(total_games):
        t1, t2 = match_list[g, 0], match_list[g, 1]
        res = outcomes[g]
        if res == 0:
            wins[t1] += 1
            losses[t2] += 1
            h2h_wins[t1, t2] += 1
            h2h_losses[t2, t1] += 1
        elif res == 2:
            wins[t2] += 1
            losses[t1] += 1
            h2h_wins[t2, t1] += 1
            h2h_losses[t1, t2] += 1
        else:
            draws[t1] += 1
            draws[t2] += 1
            h2h_draws[t1, t2] += 1
            h2h_draws[t2, t1] += 1

    win_pct = np.zeros(n_teams, dtype=np.float32)
    for i in range(n_teams):
        played = wins[i] + losses[i]
        win_pct[i] = wins[i] / played if played > 0 else 0.0

    order = np.arange(n_teams)
    for i in range(n_teams):
        for j in range(i+1, n_teams):
            ti, tj = order[i], order[j]
            if (win_pct[tj] > win_pct[ti]) or \
               (win_pct[tj] == win_pct[ti] and wins[tj] > wins[ti]) or \
               (win_pct[tj] == win_pct[ti] and wins[tj] == wins[ti] and tj < ti):
                order[i], order[j] = tj, ti

    final_order = np.empty(n_teams, dtype=np.int32)
    idx_out = 0
    i = 0
    while i < n_teams:
        same = [order[i]]
        j = i + 1
        while j < n_teams and win_pct[order[j]] == win_pct[order[i]]:
            same.append(order[j])
            j += 1

        if len(same) == 1:
            final_order[idx_out] = same[0]
            idx_out += 1
        else:
            rates = np.zeros(len(same), dtype=np.float32)
            for a in range(len(same)):
                ti = same[a]
                wsum, gsum = 0, 0
                for b in range(len(same)):
                    if a == b: continue
                    tj = same[b]
                    wsum += h2h_wins[ti, tj]
                    gsum += h2h_wins[ti, tj] + h2h_losses[ti, tj] + h2h_draws[ti, tj]
                rates[a] = wsum / gsum if gsum > 0 else 0.0

            sorted_idx = np.argsort(-rates)
            for s in sorted_idx:
                final_order[idx_out] = same[s]
                idx_out += 1
        i = j

    return final_order

# 3. 시뮬레이션 실행 함수
def run_simulation_numba(all_matches, n_jobs=-1):
    match_list = np.array([pair for day in all_matches for pair in day], dtype=np.int32)
    total_games = len(match_list)
    
    from itertools import product
    outcome_iter = product([0, 1, 2], repeat=total_games)

    rank_counter = {team: np.zeros(n_teams, dtype=np.int64) for team in teams}
    
    def simulate_one(outcome):
        out_arr = np.array(outcome, dtype=np.int32)
        result = simulate_rank_numba(
            wins0, losses0, draws0,
            h2h_wins0, h2h_losses0, h2h_draws0,
            match_list, out_arr
        )
        return result

    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_one)(outcome) for outcome in outcome_iter
    )

    for ranked_idx in results:
        for pos in range(n_teams):
            rank_counter[teams[ranked_idx[pos]]][pos] += 1

    total_cases = 3 ** total_games
    prob_dict = {team: 100* count / total_cases for team, count in rank_counter.items()}
    df = pd.DataFrame(prob_dict).T
    df.columns = [f"{i+1}위" for i in range(n_teams)]
    return df

# 4. 확률 출력 보조함수
def to_percent(df):
    return df.apply(lambda col: col.map(lambda x: f"{x:.2f}%"))
