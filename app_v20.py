# -*- coding: utf-8 -*-
"""
v20_ana 完全統合版 Streamlitアプリ
★ 期間指定・進捗ゲージ搭載・v20最適化設定版
"""
import re
import json
import concurrent.futures
import itertools
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st
import lightgbm as lgb
from bs4 import BeautifulSoup

# ============================================================
# 0. 基本設定 (Streamlit & サーバー通信設定)
# ============================================================
st.set_page_config(page_title="v20 波乱特化AI (完全統合版)", layout="wide")

JST = timezone(timedelta(hours=+9), 'JST')
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
req_session = requests.Session()
req_session.headers.update(UA)
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=3)
req_session.mount('https://', adapter)
req_session.mount('http://', adapter)

JCD_NAME = {
    1:"桐生", 2:"戸田", 3:"江戸川", 4:"平和島", 5:"多摩川", 6:"浜名湖",
    7:"蒲郡", 8:"常滑", 9:"津", 10:"三国", 11:"びわこ", 12:"住之江",
    13:"尼崎", 14:"鳴門", 15:"丸亀", 16:"児島", 17:"宮島", 18:"徳山",
    19:"下関", 20:"若松", 21:"芦屋", 22:"福岡", 23:"唐津", 24:"大村"
}

# セッション状態の初期化
if "df_results" not in st.session_state:
    st.session_state.df_results = None
if "auto_bet_queue" not in st.session_state:
    st.session_state.auto_bet_queue = []
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""

# ============================================================
# 1. モデルの読み込み
# ============================================================
@st.cache_resource
def load_models():
    try:
        model_12 = lgb.Booster(model_file='lgb_12_v20_ana.txt')
        model_13 = lgb.Booster(model_file='lgb_13_v20_ana.txt')
        return model_12, model_13
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました。詳細: {e}")
        return None, None

model_12, model_13 = load_models()

# ============================================================
# 2. データ構造・計算ロジック
# ============================================================
@dataclass
class Racer:
    name: str; age: int; cls_val: int; weight: int; f_count: int; avg_st: float
    n_win: float; n_2ren: float; l_win: float; l_2ren: float; m_2ren: float; b_2ren: float

def calc_extended_stats(racers: List[Racer]) -> List[Dict]:
    avg_win = sum(r.n_win for r in racers) / 6.0
    avg_motor = sum(r.m_2ren for r in racers) / 6.0
    avg_st = sum(r.avg_st for r in racers) / 6.0
    win_rates = sorted([r.n_win for r in racers], reverse=True)
    motors = sorted([r.m_2ren for r in racers], reverse=True)
    sts = sorted([r.avg_st for r in racers])

    stats = []
    for i, r in enumerate(racers):
        st_diff_in = round(r.avg_st - racers[i-1].avg_st, 3) if i > 0 else 0.0
        win_diff_in = round(r.n_win - racers[i-1].n_win, 2) if i > 0 else 0.0
        st_diff_out = round(r.avg_st - racers[i+1].avg_st, 3) if i < 5 else 0.0
        win_diff_out = round(r.n_win - racers[i+1].n_win, 2) if i < 5 else 0.0

        stats.append({
            "win_dev": round(r.n_win - avg_win, 2), "motor_dev": round(r.m_2ren - avg_motor, 4), "st_dev": round(avg_st - r.avg_st, 3),
            "win_rank": win_rates.index(r.n_win) + 1, "motor_rank": motors.index(r.m_2ren) + 1, "st_rank": sts.index(r.avg_st) + 1,
            "st_diff_in": st_diff_in, "win_diff_in": win_diff_in,
            "st_diff_out": st_diff_out, "win_diff_out": win_diff_out
        })
    return stats

# ============================================================
# 3. データ取得・スクレイピング
# ============================================================
def fetch_kyotei24_data(jcd, rno, date_str):
    url = f"https://info.kyotei.fun/info-{date_str}-{jcd:02d}-{rno}.html"
    try:
        r = req_session.get(url, timeout=7)
        r.encoding = r.apparent_encoding
        html = r.text if r.status_code == 200 else ""
    except: return None
    if not html or "出走表" not in html: return None

    soup = BeautifulSoup(html, "html.parser")
    
    lane_to_rank = {}
    jyuni_divs = soup.find_all('div', class_='jyuni')
    has_result = False
    if len(jyuni_divs) >= 6:
        for i in range(6):
            txt = jyuni_divs[i].get_text(strip=True)
            if txt.isdigit():
                lane_to_rank[i+1] = txt
                has_result = True
                
    payoff = 0
    actual_result = ""
    if has_result:
        ranks = {int(v): k for k, v in lane_to_rank.items()}
        if 1 in ranks and 2 in ranks and 3 in ranks:
            actual_result = f"{ranks[1]}-{ranks[2]}-{ranks[3]}"
            
        payoff_div = soup.find('div', class_='race_result_end_label', string=re.compile('3連単'))
        if payoff_div and payoff_div.parent:
            money_span = payoff_div.parent.find('span', class_='race_result_end_money_num')
            if money_span:
                ptxt = money_span.get_text(strip=True).replace(',', '')
                if ptxt.isdigit(): payoff = int(ptxt)

    rd = [{"name": f"選手{i+1}", "age": 30, "cls": 1, "weight": 50, "f": 0, "st": 0.17, "nw": 0.0, "n2": 0.0, "lw": 0.0, "l2": 0.0, "m2": 0.0, "b2": 0.0} for i in range(6)]
    cls_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    current_label = ""
    for tr in soup.find_all('tr'):
        tds = tr.find_all(['td', 'th'])
        if not tds: continue
        if len(tds) >= 7: current_label = tds[0].get_text(strip=True).replace('\n', '').replace(' ', ''); data_tds = tds[-6:]
        elif len(tds) == 6 and current_label: data_tds = tds
        else: current_label = ""; continue

        for i in range(6):
            txt = data_tds[i].get_text(" ", strip=True).replace(' ', '').replace('　', '').replace('\n', '')
            if "選手名" in current_label:
                m = re.search(r'\((\d{2})\)', txt)
                if m: rd[i]["age"] = int(m.group(1))
            elif "選手情報" in current_label or "支部" in current_label or "級" in current_label:
                m = re.search(r'([A12B]{2})', txt)
                if m: rd[i]["cls"] = cls_map.get(m.group(1), 1)
                m = re.search(r'(\d+)kg', txt, re.IGNORECASE)
                if m: rd[i]["weight"] = int(m.group(1))
            elif "全国" in current_label and "勝率" in current_label:
                m1 = re.search(r'^([\d\.]+)', txt); m2 = re.search(r'\(([\d\.]+)\)', txt)
                if m1: rd[i]["n2"] = float(m1.group(1))/100.0 if float(m1.group(1))>1.0 else float(m1.group(1))
                if m2: rd[i]["nw"] = float(m2.group(1))
            elif "当地" in current_label and "勝率" in current_label:
                m1 = re.search(r'^([\d\.]+)', txt); m2 = re.search(r'\(([\d\.]+)\)', txt)
                if m1: rd[i]["l2"] = float(m1.group(1))/100.0 if float(m1.group(1))>1.0 else float(m1.group(1))
                if m2: rd[i]["lw"] = float(m2.group(1))
            elif "モータ" in current_label and "2連率" in current_label:
                m = re.search(r'^([\d\.]+)', txt)
                if m: rd[i]["m2"] = float(m.group(1))/100.0 if float(m.group(1))>1.0 else float(m.group(1))
            elif "ボート" in current_label and "2連率" in current_label:
                m = re.search(r'^([\d\.]+)', txt)
                if m: rd[i]["b2"] = float(m.group(1))/100.0 if float(m.group(1))>1.0 else float(m.group(1))
            elif "平均ST" in current_label:
                try: rd[i]["st"] = float(txt)
                except: pass

    if sum(x["nw"] for x in rd) == 0: return None
    
    racers = []
    for x in rd:
        racers.append(Racer(
            name=x["name"], age=x["age"], cls_val=x["cls"], weight=x["weight"], f_count=x["f"],
            avg_st=x["st"], n_win=x["nw"], n_2ren=x["n2"], l_win=x["lw"], l_2ren=x["l2"],
            m_2ren=x["m2"], b_2ren=x["b2"]
        ))
    return racers, actual_result, payoff, has_result

# ============================================================
# 4. v20 スジ連動・確率計算ロジック
# ============================================================
def calculate_v20_probs(jcd, racers):
    pairs_features = []
    combinations = list(itertools.permutations([0, 1, 2, 3, 4, 5], 2))
    rel_stats = calc_extended_stats(racers)

    for a_idx, b_idx in combinations:
        a = racers[a_idx]
        b = racers[b_idx]
        a_s = rel_stats[a_idx]
        b_s = rel_stats[b_idx]
        a_lane, b_lane = a_idx + 1, b_idx + 1
        
        pairs_features.append({
            "場": jcd, 
            "A_枠": a_lane, "A_級": a.cls_val, "A_ST": a.avg_st, "A_勝率": a.n_win, "A_モータ": a.m_2ren,
            "A_勝率偏差": a_s["win_dev"], "A_内ST差": a_s["st_diff_in"], "A_外ST差": a_s["st_diff_out"],
            "B_枠": b_lane, "B_級": b.cls_val, "B_ST": b.avg_st, "B_勝率": b.n_win, "B_モータ": b.m_2ren,
            "B_勝率偏差": b_s["win_dev"], "B_内ST差": b_s["st_diff_in"], "B_外ST差": b_s["st_diff_out"],
            "枠の差": b_lane - a_lane,
            "STの差": a.avg_st - b.avg_st,
            "勝率の差": a.n_win - b.n_win
        })
        
    df_pairs = pd.DataFrame(pairs_features)
    
    features_order = [
        '場', 'A_枠', 'A_級', 'A_ST', 'A_勝率', 'A_モータ',
        'A_勝率偏差', 'A_内ST差', 'A_外ST差',
        'B_枠', 'B_級', 'B_ST', 'B_勝率', 'B_モータ',
        'B_勝率偏差', 'B_内ST差', 'B_外ST差',
        '枠の差', 'STの差', '勝率の差'
    ]
    X = df_pairs[features_order].copy()
    
    X['場'] = X['場'].astype('category')
    X['枠の差'] = X['枠の差'].astype('category')
    
    pred_12 = model_12.predict(X)
    pred_13 = model_13.predict(X)
    
    dict_12 = { (c[0]+1, c[1]+1): p for c, p in zip(combinations, pred_12) }
    dict_13 = { (c[0]+1, c[1]+1): p for c, p in zip(combinations, pred_13) }
    
    results = []
    raw_scores = []
    combos_3 = list(itertools.permutations([1, 2, 3, 4, 5, 6], 3))
    
    for a, b, c in combos_3:
        score = dict_12[(a, b)] * dict_13[(a, c)]
        raw_scores.append(score)
        
    total_score = sum(raw_scores)
    for (a, b, c), score in zip(combos_3, raw_scores):
        prob = (score / total_score) * 100 if total_score > 0 else 0
        results.append({"bet": f"{a}-{b}-{c}", "prob": round(prob, 2)})
        
    results.sort(key=lambda x: x["prob"], reverse=True)
    return results

# ============================================================
# 5. Streamlit UI & 実行処理
# ============================================================
st.sidebar.header("⚙️ 予想設定")

# ★変更点：期間で指定できるように修正（単一日の選択も可能）
date_range = st.sidebar.date_input("予想・バックテスト期間", [datetime.now(JST).date(), datetime.now(JST).date()])
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

venue_name = st.sidebar.selectbox("競艇場", ["全場一括処理"] + list(JCD_NAME.values()))
target_jcds = list(JCD_NAME.keys()) if venue_name == "全場一括処理" else [[k for k, v in JCD_NAME.items() if v == venue_name][0]]

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 買い目フィルター")
# ★変更点：v20の最適設定（バランス型）をデフォルトに指定
top_n = st.sidebar.number_input("上位何点まで買うか？ (v20推奨: 5〜6)", min_value=1, max_value=20, value=6)
# ★変更点：0.1刻みで指定できるように変更、デフォルトを1.9%に
min_prob = st.sidebar.slider("勝率の足切り下限(%) (v20推奨: 1.5〜1.9%)", min_value=0.0, max_value=50.0, value=1.9, step=0.1)
max_prob = st.sidebar.slider("勝率の足切り上限(%)", min_value=0.0, max_value=100.0, value=100.0, step=0.1)
bet_amount = st.sidebar.number_input("1点あたりの購入金額 (円)", min_value=100, max_value=10000, value=100, step=100)

def process_single_race(jcd, rno, dstr):
    res = fetch_kyotei24_data(jcd, rno, dstr)
    if not res:
        return None
        
    racers, actual_result, payoff, has_result = res
    probs = calculate_v20_probs(jcd, racers)
    
    top_n_bets = probs[:int(top_n)]
    buy_bets = [b for b in top_n_bets if min_prob <= b["prob"] <= max_prob]
    
    kaime_str = "見"
    count = 0
    hit_mark = "❌"
    result_prob_str = "-"
    actual_payoff = payoff if has_result else 0
    my_payout = 0 
    
    if buy_bets:
        kaime_str = ", ".join([f"{b['bet']}({b['prob']}%)" for b in buy_bets])
        count = len(buy_bets)
        
    if has_result and actual_result:
        res_prob = next((b["prob"] for b in probs if b["bet"] == actual_result), 0.0)
        result_prob_str = f"{actual_result}({res_prob}%)"
        
        if buy_bets:
            for b in buy_bets:
                if b["bet"] == actual_result:
                    hit_mark = "🎯"
                    my_payout = payoff
                    break
        
    return {
        "日付": dstr, "場": JCD_NAME[jcd], "R": f"{rno}R", 
        "買い目": kaime_str, "結果": actual_result if has_result else "結果待ち", 
        "結果確率": result_prob_str if has_result else "⏳", 
        "的中": hit_mark if has_result else "-", 
        "払戻金": actual_payoff, "点数": count,
        "my_payout": my_payout
    }

st.title("🚤 v20 波乱特化AI (完全統合・キューマスター対応)")

if st.button("🚀 予想 / バックテストを実行"):
    if not model_12 or not model_13:
        st.error("モデルが読み込まれていないため実行できません。")
        st.stop()

    # 期間から日数のリストを生成
    delta = end_date - start_date
    days = [(start_date + timedelta(days=i)).strftime("%Y%m%d") for i in range(delta.days + 1)]
    
    results_disp = []
    tasks = [(dstr, j, r) for dstr in days for j in target_jcds for r in range(1, 13)]
    total_tasks = len(tasks)
    
    # ★変更点：進捗ゲージ（プログレスバー）の追加
    st.write(f"🔍 全 {total_tasks} レースのデータを取得・解析します...")
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        future_to_task = {executor.submit(process_single_race, j, r, dstr): (dstr, j, r) for dstr, j, r in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            res_val = future.result()
            if res_val is not None:
                results_disp.append(res_val)
            
            done += 1
            # ゲージとテキストをリアルタイム更新
            progress_bar.progress(done / total_tasks)
            progress_text.text(f"⚡ 解析進捗: {done} / {total_tasks} レース完了")
            
    if results_disp:
        df = pd.DataFrame(results_disp)
        df['場_id'] = df['場'].map({v: k for k, v in JCD_NAME.items()})
        df['R_num'] = df['R'].str.replace('R', '').astype(int)
        df = df.sort_values(by=['日付', '場_id', 'R_num']).drop(columns=['場_id', 'R_num']).reset_index(drop=True)
        
        total_bets = df["点数"].sum()
        total_cost = total_bets * bet_amount
        total_payoff = df["my_payout"].sum() * (bet_amount / 100)
        roi = (total_payoff / total_cost * 100) if total_cost > 0 else 0
        
        st.session_state.summary_text = f"✅ 処理完了！ 購入点数: {total_bets}点 (投資: {total_cost:,.0f}円) / 回収: {total_payoff:,.0f}円 / 回収率: {roi:.1f}%"
        st.session_state.df_results = df
        
        # キューマスター用一時データの作成
        auto_bet_queue = []
        for index, row in df.iterrows():
            if row["点数"] > 0 and row["買い目"] != "見" and row["買い目"] != "データなし":
                bets = row["買い目"].split(", ")
                for bet_str in bets:
                    try:
                        clean_bet = bet_str.split("(")[0]
                        pattern = [int(x) for x in clean_bet.split("-")]
                        queue_item = {
                            "venue": row["場"],
                            "race": str(row["R"]).replace("R", ""),
                            "pattern": pattern,
                            "amount": str(bet_amount)
                        }
                        auto_bet_queue.append(queue_item)
                    except:
                        pass
        st.session_state.auto_bet_queue = auto_bet_queue
    else:
        st.session_state.df_results = None
        st.session_state.auto_bet_queue = []
        st.session_state.summary_text = ""
        st.warning("指定した期間・条件で開催されているレースが見つかりませんでした。")

# 結果の描画
if st.session_state.df_results is not None:
    st.success(st.session_state.summary_text)
    disp_cols = ["日付", "場", "R", "買い目", "結果", "結果確率", "的中", "払戻金"]
    st.dataframe(st.session_state.df_results[disp_cols], use_container_width=True)

    if st.session_state.auto_bet_queue:
        st.markdown("---")
        st.subheader("🤖 全自動購入用データ (キューマスター専用)")
        if st.button("キューマスター用JSONデータを生成"):
            json_string = json.dumps(st.session_state.auto_bet_queue, ensure_ascii=False, indent=4)
            st.caption("右上のコピーボタンを押して、キューマスターに貼り付けてください。")
            st.code(json_string, language="json")
    else:
        st.info("条件に合致する買い目がありませんでした。設定（閾値など）を見直してください。")
