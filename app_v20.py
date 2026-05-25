# -*- coding: utf-8 -*-
"""
v20_ana 新AI対応 Streamlitアプリ (高速スクレイピング版)
1着-2着スジ ＆ 1着-3着スジ の連動確率を掛け合わせる3連単予想AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import lightgbm as lgb
import itertools
import re
import concurrent.futures

st.set_page_config(page_title="v20 波乱特化AI (スジ展開・高速版)", layout="wide")

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
        st.error(f"モデルの読み込みに失敗しました。\n詳細: {e}")
        return None, None

model_12, model_13 = load_models()

# ============================================================
# 2. 高速スクレイピング設定
# ============================================================
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
req_session = requests.Session()
req_session.headers.update(UA)
adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=3)
req_session.mount('https://', adapter)
req_session.mount('http://', adapter)

JCD_NAME = {
    1:"桐生", 2:"戸田", 3:"江戸川", 4:"平和島", 5:"多摩川", 6:"浜名湖",
    7:"蒲郡", 8:"常滑", 9:"津", 10:"三国", 11:"びわこ", 12:"住之江",
    13:"尼崎", 14:"鳴門", 15:"丸亀", 16:"児島", 17:"宮島", 18:"徳山",
    19:"下関", 20:"若松", 21:"芦屋", 22:"福岡", 23:"唐津", 24:"大村"
}

def fetch_race_data(jcd, rno, date_str):
    url = f"https://info.kyotei.fun/info-{date_str}-{jcd:02d}-{rno}.html"
    try:
        r = req_session.get(url, timeout=5)
        r.encoding = r.apparent_encoding
        html = r.text if r.status_code == 200 else ""
    except: return None
    if not html or "出走表" not in html: return None

    soup = BeautifulSoup(html, "html.parser")
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
    return rd

# ============================================================
# 3. 確率計算ロジック（1-2スジ × 1-3スジ の融合）
# ============================================================
def calculate_v20_probs(jcd, racers):
    pairs_features = []
    combinations = list(itertools.permutations([0, 1, 2, 3, 4, 5], 2))
    
    # 全30通りのペア（2連単）の特徴量を生成
    for a_idx, b_idx in combinations:
        a = racers[a_idx]
        b = racers[b_idx]
        a_lane = a_idx + 1
        b_lane = b_idx + 1
        
        pairs_features.append({
            "場": jcd, 
            "A_枠": a_lane, "A_級": a["cls"], "A_ST": a["st"], "A_勝率": a["nw"], "A_モータ": a["m2"],
            "B_枠": b_lane, "B_級": b["cls"], "B_ST": b["st"], "B_勝率": b["nw"], "B_モータ": b["m2"],
            "枠の差": b_lane - a_lane,
            "STの差": a["st"] - b["st"],
            "勝率の差": a["nw"] - b["nw"]
        })
        
    df_pairs = pd.DataFrame(pairs_features)
    df_pairs['場'] = df_pairs['場'].astype('category')
    df_pairs['枠の差'] = df_pairs['枠の差'].astype('category')
    
    # モデルで「1着-2着の確率」と「1着-3着の確率」をそれぞれ予測
    pred_12 = model_12.predict(df_pairs)
    pred_13 = model_13.predict(df_pairs)
    
    # ペア（A, B）をキーにして確率を取得できる辞書を作成
    dict_12 = { (comb[0]+1, comb[1]+1): pred for comb, pred in zip(combinations, pred_12) }
    dict_13 = { (comb[0]+1, comb[1]+1): pred for comb, pred in zip(combinations, pred_13) }
    
    # 3連単（全120通り）の確率を計算
    results = []
    total_score = 0
    raw_scores = []
    
    combos_3 = list(itertools.permutations([1, 2, 3, 4, 5, 6], 3))
    for a, b, c in combos_3:
        # A-Bの1-2着スジ確率 × A-Cの1-3着スジ確率 を掛け合わせる
        score = dict_12[(a, b)] * dict_13[(a, c)]
        raw_scores.append(score)
        total_score += score
        
    for (a, b, c), score in zip(combos_3, raw_scores):
        prob = (score / total_score) * 100
        results.append({
            "bet": f"{a}-{b}-{c}",
            "prob": round(prob, 2)
        })
        
    results.sort(key=lambda x: x["prob"], reverse=True)
    return results

# ============================================================
# 4. 並列処理用の関数
# ============================================================
def process_single_race(jcd, rno, date_str, top_n, min_prob, max_prob):
    racers = fetch_race_data(jcd, rno, date_str)
    if racers is None:
        return {"R": f"{rno}R", "買い目": "データなし", "点数": 0}
        
    probs = calculate_v20_probs(jcd, racers)
    
    # ユーザー設定に従って買い目を絞り込む
    top_n_bets = probs[:int(top_n)]
    buy_bets = [b for b in top_n_bets if min_prob <= b["prob"] <= max_prob]
    
    if buy_bets:
        kaime_str = ", ".join([f"{b['bet']}({b['prob']}%)" for b in buy_bets])
        count = len(buy_bets)
    else:
        kaime_str = "見"
        count = 0
        
    return {"R": f"{rno}R", "買い目": kaime_str, "点数": count}

# ============================================================
# 5. Streamlit UI
# ============================================================
st.title("🚤 v20_ana 新AI予想（スジ学習・高速並列版）")

st.sidebar.header("⚙️ 予想設定")
target_date = st.sidebar.date_input("予想する日付", datetime.now())
date_str = target_date.strftime("%Y%m%d")

venue_name = st.sidebar.selectbox("競艇場", list(JCD_NAME.values()))
jcd = [k for k, v in JCD_NAME.items() if v == venue_name][0]

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 買い目フィルター")
top_n = st.sidebar.number_input("上位何点まで買うか？ (推奨: 1点)", min_value=1, max_value=20, value=1)
min_prob = st.sidebar.slider("勝率の足切り下限(%) (推奨: 11%)", min_value=0.0, max_value=50.0, value=11.0, step=1.0)
max_prob = st.sidebar.slider("勝率の足切り上限(%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)

if st.button(f"{venue_name} の予想を取得"):
    if not model_12 or not model_13:
        st.error("モデルが読み込まれていないため実行できません。")
        st.stop()

    with st.spinner(f"⚡ {venue_name} の全12レースを高速同時取得中..."):
        results_disp = []
        
        # 🚀 concurrent.futures を用いた 12レース並列取得処理
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            future_to_rno = {executor.submit(process_single_race, jcd, rno, date_str, top_n, min_prob, max_prob): rno for rno in range(1, 13)}
            
            # 結果をレース順に並べるための一時辞書
            temp_results = {}
            for future in concurrent.futures.as_completed(future_to_rno):
                rno = future_to_rno[future]
                temp_results[rno] = future.result()
                
        # 1R〜12Rの順番に並び替えてリスト化
        for rno in range(1, 13):
            if rno in temp_results:
                results_disp.append(temp_results[rno])
            
    df_results = pd.DataFrame(results_disp)
    st.dataframe(df_results, use_container_width=True)
    
    total_bets = df_results["点数"].sum()
    st.info(f"💰 今日の合計購入点数: {total_bets}点 (1点100円なら {total_bets * 100}円)")
