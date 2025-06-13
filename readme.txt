# KBO 시뮬레이터

KBO 리그의 남은 경기 결과에 따른 전체 순위를 시뮬레이션합니다.
NumPy + Numba + Streamlit 기반으로 제작되었고, 웹 브라우저에서 쉽게 사용할 수 있습니다.

## 실행 방법

1. 의존 패키지 설치:

(bash)

pip install -r requirements.txt


이 프로젝트는 KBO(한국프로야구)의 남은 경기 결과에 따라 전체 팀 순위가 어떻게 변화할 수 있는지를 시뮬레이션하는 도구입니다.  
NumPy, Numba, pandas를 활용하여 빠르고 정확하게 결과를 예측하며, Streamlit을 통해 웹 인터페이스로 제공합니다.

---

## 주요 기능

- 전체 순위 시뮬레이션: 남은 경기의 모든 승/무/패 조합을 계산하여 팀별 순위 확률을 예측
- 상대전적 tie-breaker 반영: 동일 승률일 경우 상대전적에 따라 순위 조정
- Streamlit 기반 웹 UI: 브라우저에서 직접 시뮬레이션 실행
- 빠른 계산 속도: Numba로 연산 최적화, joblib을 통한 병렬 처리

---

## 사용된 기술 스택

- Python 3.11+
- pandas
- numpy
- numba
- joblib
- streamlit
- lxml (웹 데이터 파싱용)

---

## 사용 방법

1. bash (터미널, 파워쉘 등) 을 열고 아래 코드를 입력합니다.

git clone https://github.com/MSILJI0708/kbo-simulator.git

cd kbo-simulator


2. streamlit run app.py 를 입력하면, 자동으로 사이트가 열립니다.

3. app.py 에서 설정된 매치업대로 시뮬레이션 경기를 진행합니다. 매치업의 사이즈에 따라 작업 시간은 상이할 수 있습니다.