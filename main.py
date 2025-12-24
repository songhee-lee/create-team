import streamlit as st
import random
from itertools import combinations
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="Joshua Team Generator",
    page_icon="👥",
    layout="wide"
)

# secrets에서 기본값 로드 (있으면)
def load_defaults_from_secrets():
    """secrets.toml에서 기본값 로드"""
    defaults = {
        'names': None,
        'n_people': 12,
        'num_teams': 4,
        'distribution_type': '균등'
    }
    
    try:
        if 'people' in st.secrets and 'names' in st.secrets['people']:
            defaults['names'] = list(st.secrets['people']['names'])
            # secrets에 이름이 있으면 이름 개수를 우선순위로 사용
            defaults['n_people'] = len(defaults['names'])
        
        if 'default' in st.secrets:
            # n_people은 이름이 없을 때만 적용
            if 'n_people' in st.secrets['default'] and not defaults['names']:
                defaults['n_people'] = st.secrets['default']['n_people']
            if 'num_teams' in st.secrets['default']:
                defaults['num_teams'] = st.secrets['default']['num_teams']
            # 하위 호환성을 위해 team_size도 확인
            elif 'team_size' in st.secrets['default']:
                # team_size가 있으면 num_teams로 변환 (대략적으로)
                defaults['num_teams'] = max(1, defaults['n_people'] // st.secrets['default']['team_size'])
    except Exception as e:
        # secrets 파일이 없거나 오류가 있으면 기본값 사용
        pass
    
    return defaults

# 기본값 로드
_defaults = load_defaults_from_secrets()

# 세션 상태 초기화
if 'rounds' not in st.session_state:
    st.session_state.rounds = []
if 'meeting_count' not in st.session_state:
    st.session_state.meeting_count = defaultdict(int)
if 'people_names' not in st.session_state:
    if _defaults['names']:
        st.session_state.people_names = _defaults['names'].copy()
    else:
        st.session_state.people_names = []
if 'n_people' not in st.session_state:
    st.session_state.n_people = _defaults['n_people']
if 'num_teams' not in st.session_state:
    st.session_state.num_teams = _defaults['num_teams']
if 'duplicate_people' not in st.session_state:
    st.session_state.duplicate_people = {}

def get_meeting_score(team, meeting_count):
    """팀 구성원들이 이미 만난 횟수의 합을 계산"""
    score = 0
    for i in range(len(team)):
        for j in range(i + 1, len(team)):
            pair = tuple(sorted([team[i], team[j]]))
            score += meeting_count[pair]
    return score

def update_meetings(team, meeting_count):
    """팀 구성원들의 만남 기록 업데이트"""
    for i in range(len(team)):
        for j in range(i + 1, len(team)):
            pair = tuple(sorted([team[i], team[j]]))
            meeting_count[pair] += 1

def create_round_greedy(n_people, team_size, meeting_count, team_distribution=None):
    """Greedy 알고리즘으로 한 라운드의 팀 구성
    
    Args:
        n_people: 전체 인원 수
        team_size: 기본 팀 크기
        meeting_count: 만남 횟수 기록
        team_distribution: 팀 크기 분포 리스트 (예: [5, 5, 5, 6] - 5명 팀 3개, 6명 팀 1개)
    """
    teams = []
    remaining = list(range(n_people))
    random.shuffle(remaining)
    
    # 팀 분포가 지정된 경우
    if team_distribution:
        for target_size in team_distribution:
            if len(remaining) < target_size:
                break
                
            best_team = None
            best_score = float('inf')
            
            # 가능한 팀 조합 검토
            if len(remaining) <= 15:
                possible_teams = list(combinations(remaining, target_size))
            else:
                # 많은 경우 랜덤 샘플링
                possible_teams = []
                for _ in range(min(1000, len(list(combinations(remaining, target_size))))):
                    team = random.sample(remaining, target_size)
                    possible_teams.append(tuple(team))
            
            # 가장 적게 만난 조합 선택
            for team in possible_teams:
                score = get_meeting_score(team, meeting_count)
                if score < best_score:
                    best_score = score
                    best_team = team
            
            if best_team:
                teams.append(list(best_team))
                for person in best_team:
                    remaining.remove(person)
                update_meetings(best_team, meeting_count)
    else:
        # 기존 로직 (모든 팀 동일 크기)
        while len(remaining) >= team_size:
            best_team = None
            best_score = float('inf')
            
            # 가능한 팀 조합 검토
            if len(remaining) <= 15:
                possible_teams = list(combinations(remaining, team_size))
            else:
                # 많은 경우 랜덤 샘플링
                possible_teams = []
                for _ in range(min(1000, len(list(combinations(remaining, team_size))))):
                    team = random.sample(remaining, team_size)
                    possible_teams.append(tuple(team))
            
            # 가장 적게 만난 조합 선택
            for team in possible_teams:
                score = get_meeting_score(team, meeting_count)
                if score < best_score:
                    best_score = score
                    best_team = team
            
            if best_team:
                teams.append(list(best_team))
                for person in best_team:
                    remaining.remove(person)
                update_meetings(best_team, meeting_count)
    
    # 남은 사람들 처리
    if remaining:
        teams.append(remaining)
        if len(remaining) >= 2:
            update_meetings(remaining, meeting_count)
    
    return teams

def calculate_team_distribution(n_people, num_teams):
    """팀 개수 기준으로 팀 크기 분포 계산
    
    Args:
        n_people: 전체 인원 수
        num_teams: 팀 개수
    
    Returns:
        list: 각 팀의 인원 수 리스트 (예: [4, 4, 3])
    """
    if num_teams > n_people:
        num_teams = n_people
    
    # 기본 팀 크기와 남은 인원
    base_size = n_people // num_teams
    remainder = n_people % num_teams
    
    # 큰 팀 개수 = remainder, 작은 팀 개수 = num_teams - remainder
    distribution = [base_size + 1] * remainder + [base_size] * (num_teams - remainder)
    
    return distribution

def find_duplicate_pairs(current_round, previous_rounds):
    """현재 라운드에서 이전 라운드와 중복되는 쌍 찾기
    
    Returns:
        dict: {person_id: [중복된 상대방들]} 형태
    """
    # 이전 라운드의 모든 쌍 수집
    previous_pairs = set()
    for round_teams in previous_rounds:
        for team in round_teams:
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    pair = tuple(sorted([team[i], team[j]]))
                    previous_pairs.add(pair)
    
    # 현재 라운드에서 중복된 쌍 찾기
    duplicate_people = {}
    for team in current_round:
        for i in range(len(team)):
            for j in range(i + 1, len(team)):
                pair = tuple(sorted([team[i], team[j]]))
                if pair in previous_pairs:
                    # 중복 발견
                    if team[i] not in duplicate_people:
                        duplicate_people[team[i]] = []
                    if team[j] not in duplicate_people:
                        duplicate_people[team[j]] = []
                    
                    duplicate_people[team[i]].append(team[j])
                    duplicate_people[team[j]].append(team[i])
    
    return duplicate_people

def create_meeting_heatmap(n_people, meeting_count, people_names):
    """만남 횟수 히트맵 생성"""
    # 매트릭스 생성
    matrix = np.zeros((n_people, n_people))
    
    for (i, j), count in meeting_count.items():
        if i < n_people and j < n_people:  # 안전장치
            matrix[i][j] = count
            matrix[j][i] = count
    
    # 이름 리스트 안전하게 생성
    safe_names = []
    for i in range(n_people):
        if i < len(people_names):
            safe_names.append(people_names[i])
        else:
            safe_names.append(f"사람{i+1}")
    
    # Plotly 히트맵 생성
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=safe_names,
        y=safe_names,
        colorscale='RdYlGn_r',
        text=matrix,
        texttemplate='%{text:.0f}',
        textfont={"size": 10},
        hovertemplate='%{y} ↔ %{x}<br>만남 횟수: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='팀원 간 만남 횟수',
        xaxis_title='',
        yaxis_title='',
        height=min(600, max(400, n_people * 30)),
        width=min(800, max(400, n_people * 30))
    )
    
    return fig

def create_round_stats_chart(rounds, meeting_count):
    """라운드별 통계 차트 생성"""
    if not rounds:
        return None
    
    round_stats = []
    cumulative_pairs = set()
    
    for round_idx, teams in enumerate(rounds, 1):
        # 이번 라운드의 쌍들
        current_pairs = set()
        duplicate_count = 0
        
        for team in teams:
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    pair = tuple(sorted([team[i], team[j]]))
                    current_pairs.add(pair)
                    
                    # 중복 체크
                    if pair in cumulative_pairs:
                        duplicate_count += 1
        
        cumulative_pairs.update(current_pairs)
        
        round_stats.append({
            '라운드': f'R{round_idx}',
            '새로운 쌍': len(current_pairs) - duplicate_count,
            '중복된 쌍': duplicate_count
        })
    
    df = pd.DataFrame(round_stats)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='새로운 쌍',
        x=df['라운드'],
        y=df['새로운 쌍'],
        marker_color='#667eea'
    ))
    
    fig.add_trace(go.Bar(
        name='중복된 쌍',
        x=df['라운드'],
        y=df['중복된 쌍'],
        marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        title='라운드별 팀 구성 분석',
        xaxis_title='라운드',
        yaxis_title='쌍의 개수',
        barmode='stack',
        height=400
    )
    
    return fig

def create_person_meeting_chart(n_people, meeting_count, people_names):
    """각 사람별 만남 횟수 차트"""
    person_counts = [0] * n_people
    
    for (i, j), count in meeting_count.items():
        if i < n_people:  # 안전장치
            person_counts[i] += count
        if j < n_people:  # 안전장치
            person_counts[j] += count
    
    # 이름 리스트 안전하게 생성
    safe_names = []
    for i in range(n_people):
        if i < len(people_names):
            safe_names.append(people_names[i])
        else:
            safe_names.append(f"사람{i+1}")
    
    df = pd.DataFrame({
        '이름': safe_names,
        '만남 횟수': person_counts
    })
    
    df = df.sort_values('만남 횟수', ascending=True)
    
    fig = px.bar(
        df,
        y='이름',
        x='만남 횟수',
        orientation='h',
        title='각 사람별 총 만남 횟수',
        color='만남 횟수',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=max(400, n_people * 25))
    
    return fig

def create_team_size_distribution(latest_round):
    """현재 라운드의 팀 크기 분포 차트"""
    team_sizes = [len(team) for team in latest_round]
    size_counts = {}
    
    for size in team_sizes:
        size_counts[size] = size_counts.get(size, 0) + 1
    
    df = pd.DataFrame([
        {'팀 크기': f'{size}명', '팀 수': count}
        for size, count in sorted(size_counts.items())
    ])
    
    fig = px.pie(
        df,
        values='팀 수',
        names='팀 크기',
        title='현재 라운드 팀 크기 분포',
        color_discrete_sequence=px.colors.sequential.Purples_r
    )
    
    return fig

def reset_state():
    """상태 초기화"""
    st.session_state.rounds = []
    st.session_state.meeting_count = defaultdict(int)

def generate_new_round(n_people, num_teams):
    """새로운 라운드 생성"""
    team_dist = calculate_team_distribution(n_people, num_teams)
    
    new_teams = create_round_greedy(n_people, 0, st.session_state.meeting_count, team_dist)
    st.session_state.rounds.append(new_teams)
    
    # 중복 확인
    if len(st.session_state.rounds) > 1:
        st.session_state.duplicate_people = find_duplicate_pairs(
            new_teams, 
            st.session_state.rounds[:-1]
        )
    else:
        st.session_state.duplicate_people = {}
    
    return True

# 타이틀
st.title("👥 Joshua Team Generator")
st.markdown("---")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    n_people = st.number_input(
        "전체 인원 수 (N)", 
        min_value=3, 
        max_value=100, 
        value=_defaults['n_people'],
        step=1
    )
    
    num_teams = st.number_input(
        "팀 개수 (M)", 
        min_value=1, 
        max_value=n_people, 
        value=min(_defaults['num_teams'], n_people),
        step=1,
        help="전체 인원을 몇 개의 팀으로 나눌지 설정합니다"
    )
    
    # 팀 분포 미리보기
    st.markdown("---")
    st.subheader("📊 팀 구성 미리보기")
    
    team_dist = calculate_team_distribution(n_people, num_teams)
    dist_desc = " + ".join([f"{size}명" for size in team_dist])
    st.markdown(f"**{num_teams}개 팀:** `{dist_desc}`")
    
    # 설정이 변경되었는지 확인
    if st.session_state.n_people != n_people or st.session_state.num_teams != num_teams:
        st.session_state.n_people = n_people
        st.session_state.num_teams = num_teams
        # 이름 리스트 초기화
        if len(st.session_state.people_names) != n_people:
            # secrets에 이름이 있고 인원수가 맞으면 사용
            if _defaults['names'] and len(_defaults['names']) == n_people:
                st.session_state.people_names = _defaults['names'].copy()
            # secrets 이름이 있지만 인원수가 다르면
            elif _defaults['names']:
                # secrets 이름을 최대한 사용하고, 부족하면 "사람X" 추가
                base_names = _defaults['names'].copy()
                if len(base_names) > n_people:
                    st.session_state.people_names = base_names[:n_people]
                else:
                    st.session_state.people_names = base_names + [f"사람{i+1}" for i in range(len(base_names), n_people)]
            # secrets에 이름이 없으면 기본 형식 사용
            else:
                st.session_state.people_names = [f"사람{i+1}" for i in range(n_people)]
    
    st.markdown("---")
    
    # 이름 입력 섹션
    st.subheader("👤 인원 이름 입력")
    
    with st.expander("이름 편집하기", expanded=False):
        for i in range(n_people):
            st.session_state.people_names[i] = st.text_input(
                f"사람 {i+1}",
                value=st.session_state.people_names[i],
                key=f"name_{i}"
            )
    
    st.markdown("---")
    
    # 통계
    st.subheader("📊 현황")
    st.metric("생성된 라운드", f"{len(st.session_state.rounds)}개")

# 메인 영역
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🎲 라운드 생성", type="primary", use_container_width=True):
        generate_new_round(n_people, num_teams)
        st.success(f"✅ 라운드 {len(st.session_state.rounds)} 생성 완료!")

with col2:
    if st.button("🔄 리셋", type="secondary", use_container_width=True):
        reset_state()
        st.rerun()

st.markdown("---")

# 생성된 라운드 표시 (수정된 버전)
if st.session_state.rounds:
    # 가장 최근 라운드
    latest_round = st.session_state.rounds[-1]
    duplicate_people = st.session_state.duplicate_people
    
    st.subheader(f"🎯 라운드 {len(st.session_state.rounds)}")
    
    # 중복 경고 메시지
    if duplicate_people:
        st.warning(f"⚠️ {len(duplicate_people)}명이 이전 라운드와 중복된 팀원과 함께합니다")
    
    # CSS 스타일 적용
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
        
        .team-box {
            font-family: 'Noto Sans KR', sans-serif;
            border: 2px solid #e9ecef;
            border-radius: 12px;
            padding: 12px;
            background: #ffffff;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 12px;
        }
        
        .team-header {
            font-size: 14px;
            font-weight: 500;
            color: #495057;
            margin-bottom: 10px;
            text-align: center;
            border-bottom: 1px solid #e9ecef;
            padding-bottom: 8px;
        }
        
        .team-count {
            color: #868e96;
            font-weight: 400;
        }
        
        .member-row {
            display: flex;
            gap: 6px;
            margin-bottom: 6px;
        }
        
        .member-card {
            flex: 1;
            padding: 8px 10px;
            border-radius: 6px;
            text-align: center;
            font-size: 13px;
            font-weight: 400;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        
        .member-normal {
            background: #f8f9fa;
            color: #2d3436;
        }
        
        .member-duplicate {
            background: #ff6b6b;
            color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 4개씩 그룹으로 나누어 표시
    teams_per_row = 4
    num_teams = len(latest_round)
    
    for row_start in range(0, num_teams, teams_per_row):
        # 4분할 컬럼 생성
        cols = st.columns(teams_per_row)
        
        for col_idx in range(teams_per_row):
            team_idx = row_start + col_idx
            
            if team_idx < num_teams:
                team = latest_round[team_idx]
                
                with cols[col_idx]:
                    # HTML 문자열 빌드
                    html = '<div class="team-box">'
                    html += f'<div class="team-header">팀 {team_idx + 1} <span class="team-count">({len(team)}명)</span></div>'
                    
                    # 팀원들을 2열로 배치
                    for i in range(0, len(team), 2):
                        html += '<div class="member-row">'
                        
                        # 왼쪽 카드
                        person_id = team[i]
                        is_duplicate = person_id in duplicate_people
                        card_class = "member-duplicate" if is_duplicate else "member-normal"
                        name = st.session_state.people_names[person_id]
                        html += f'<div class="member-card {card_class}">{name}</div>'
                        
                        # 오른쪽 카드 (있으면)
                        if i + 1 < len(team):
                            person_id = team[i + 1]
                            is_duplicate = person_id in duplicate_people
                            card_class = "member-duplicate" if is_duplicate else "member-normal"
                            name = st.session_state.people_names[person_id]
                            html += f'<div class="member-card {card_class}">{name}</div>'
                        else:
                            html += '<div style="flex: 1;"></div>'
                        
                        html += '</div>'
                    
                    html += '</div>'
                    
                    # HTML 렌더링
                    st.markdown(html, unsafe_allow_html=True)
    
    # 이전 라운드 히스토리
    if len(st.session_state.rounds) > 1:
        st.markdown("---")
        
        with st.expander("📜 이전 라운드 보기", expanded=False):
            for round_idx, round_teams in enumerate(st.session_state.rounds[:-1]):
                st.markdown(f"**라운드 {round_idx + 1}**")
                for team_idx, team in enumerate(round_teams):
                    team_names = [st.session_state.people_names[p] for p in team]
                    st.markdown(f"  - 팀 {team_idx + 1} ({len(team)}명): {', '.join(team_names)}")
                st.markdown("")

else:
    st.info("👆 '라운드 생성' 버튼을 눌러 첫 번째 라운드를 만들어보세요!")


# 하단 정보
st.markdown("---")

# 시각화 섹션 (라운드가 생성된 경우에만 표시)
if st.session_state.rounds:
    st.markdown("#### 📊 통계 및 시각화")
    
    # 탭 생성
    viz_tab1, viz_tab2 = st.tabs([
        "라운드별 분석", 
        "만남 히트맵"
    ])
    
    with viz_tab1:
        st.markdown("각 라운드에서 새롭게 만난 쌍과 이전에 만났던 쌍(중복)의 수를 보여줍니다.")
        
        fig_rounds = create_round_stats_chart(st.session_state.rounds, st.session_state.meeting_count)
        if fig_rounds:
            st.plotly_chart(fig_rounds, use_container_width=True)
        
        # 전체 통계
        total_pairs = n_people * (n_people - 1) // 2
        met_pairs = len([v for v in st.session_state.meeting_count.values() if v > 0])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("전체 가능한 쌍", f"{total_pairs}개")
        with col2:
            st.metric("만난 쌍", f"{met_pairs}개")
        with col3:
            coverage = (met_pairs / total_pairs * 100) if total_pairs > 0 else 0
            st.metric("커버리지", f"{coverage:.1f}%")
        with col4:
            avg_meetings = sum(st.session_state.meeting_count.values()) / len(st.session_state.meeting_count) if st.session_state.meeting_count else 0
            st.metric("평균 만남", f"{avg_meetings:.2f}회")
    
    with viz_tab2:
        st.markdown("각 사람이 서로 몇 번 같은 팀이 되었는지 보여줍니다. 숫자가 클수록 자주 만난 것입니다.")
        
        fig_heatmap = create_meeting_heatmap(n_people, st.session_state.meeting_count, st.session_state.people_names)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # 가장 많이 만난 쌍
        if st.session_state.meeting_count:
            max_meetings = max(st.session_state.meeting_count.values())
            most_met = [(i, j, count) for (i, j), count in st.session_state.meeting_count.items() if count == max_meetings]
            
            if most_met and max_meetings > 0:
                st.markdown(f"**가장 많이 만난 쌍 ({max_meetings}회):**")
                for i, j, count in most_met[:5]:  # 상위 5개만
                    st.write(f"- {st.session_state.people_names[i]} ↔ {st.session_state.people_names[j]}")
    

st.markdown("---")

# 범례
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; font-family: 'Noto Sans KR', sans-serif;">
            <div style="
                width: 60px;
                height: 30px;
                background: #f8f9fa;
                border-radius: 6px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 11px;
                color: #2d3436;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            ">이름</div>
            <span style="color: #495057; font-size: 14px;">정상 배정</span>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; font-family: 'Noto Sans KR', sans-serif;">
            <div style="
                width: 60px;
                height: 30px;
                background: #ff6b6b;
                border-radius: 6px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 11px;
                color: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            ">이름</div>
            <span style="color: #495057; font-size: 14px;">이전 라운드 팀원과 중복</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 14px;">
    💡 <strong>사용법:</strong> '라운드 생성' 버튼을 누르면 전체 인원이 설정한 개수의 팀으로 나뉩니다.<br>
    다시 버튼을 누르면 이전 라운드와 팀 구성이 겹치지 않는 새로운 라운드가 생성됩니다.<br>
    '리셋' 버튼을 누르면 모든 기록이 초기화됩니다.<br>
    <strong>빨간색</strong>으로 표시된 사람은 이전 라운드에서 같은 팀이었던 사람과 다시 만났습니다.
    </div>
    """,
    unsafe_allow_html=True
)
