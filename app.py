# app.py
import streamlit as st
import time

from simulator import name_to_idx, convert_match_teamnames_to_indices, run_simulation_numba, to_percent

st.title("KBO 순위 시뮬레이터")
st.markdown(" 경기 결과에 따른 순위 확률을 시뮬레이션합니다.")




# 예시 경기 선택
example_games = [
    ["LG", "SSG"], ["KIA", "두산"], ["삼성", "한화"], ["롯데", "NC"], ["KT", "키움"]
]
st.write("### 오늘의 경기")



for g in example_games:
    st.write(f"- {g[0]} vs {g[1]}")

if st.button("시뮬레이션 시작"):
    start = time.time()
    all_matches = convert_match_teamnames_to_indices([example_games], name_to_idx)
    result = run_simulation_numba(all_matches, n_jobs=-1)
    st.write("### 결과 확률 (%)")
    st.dataframe(to_percent(result))
    st.success(f"소요 시간: {time.time() - start:.2f}초")