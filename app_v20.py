# -*- coding: utf-8 -*-
"""
v20 全艇スコア解析アプリ（中穴〜大穴特化・スジ連動AI 搭載版）
※使い勝手はv19をそのまま踏襲し、内部の予測ロジックをv20へアップデート
"""
import re
import json
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import lightgbm as lgb
from bs4 import BeautifulSoup
import itertools

st.set_page_config(page_title="v20 波乱特化AI (スジ展開・キューマスター対応)", layout="wide")

JST = timezone(timedelta(hours=+9), 'JST')
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

# ============================================================
# 1. モデルの読み込み (v20 スジ連動モデル)
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
# 2. データ取得・スクレイピング (結果取得対応)
# ============================================================
def fetch_kyotei24_data(jcd, rno, date_str):
    url = f"https://info.kyotei.fun/info-{date_str}-{jcd:02d}-{rno}.html"
    try:
        r = req_session.get(url, timeout=5)
        r.encoding = r.apparent_encoding
        html = r.text if r.status_code == 200 else ""
    except: return None
    if not html or "出走表" not in html: return None

    soup = BeautifulSoup(html, "html.parser")
    
    # --- 結果の取得（バックテスト用） ---
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
        # 1着-2着-3着の文字列作成
        ranks = {int(v): k for k, v in lane_to_rank.items()}
        if 1 in ranks and 2 in ranks and 3 in ranks:
            actual_result = f"{ranks[1]}-{ranks[2]}-{ranks[3]}"
            
        payoff_div = soup.find('div', class_='race_result_end_label', string=re.compile('3連単'))
        if payoff_div and payoff_div.parent:
            money_span = payoff_div.parent.find('span', class_='race_result_end_money_num')
            if money_span:
                ptxt = money_span.get_text(strip=True).replace(',', '')
                if ptxt.isdigit(): payoff = int(ptxt)

    # --- 出走表データの取得 ---
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
    return rd, actual_result, payoff, has_result

# ============================================================
# 3. v20 確率計算ロジック（スジ連動）
# ============================================================
def calculate_v20_probs(jcd, racers):
    pairs_features = []
    combinations = list(itertools.permutations([0, 1, 2, 3, 4, 5], 2))
    
    for a_idx, b_idx in combinations:
        a, b = racers[a_idx], racers[b_idx]
        a_lane, b_lane = a_idx + 1, b_idx + 1
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
    
    pred_12 = model_12.predict(df_pairs)
    pred_13 = model_13.predict(df_pairs)
    
    dict_12 = { (c[0]+1, c[1]+1): p for c, p in zip(combinations, pred_12) }
    dict_13 = { (c[0]+1, c[1]+1): p for c, p in zip(combinations, pred_13) }
    
    results = []
    raw_scores = []
    combos_3 = list(itertools.permutations([1, 2, 3, 4, 5, 6], 3))
    
    for a, b, c in combos_3:
        # v20スジロジック：1-2着確率 × 1-3着確率
        score = dict_12[(a, b)] * dict_13[(a, c)]
        raw_scores.append(score)
        
    total_score = sum(raw_scores)
    for (a, b, c), score in zip(combos_3, raw_scores):
        prob = (score / total_score) * 100 if total_score > 0 else 0
        results.append({"bet": f"{a}-{b}-{c}", "prob": round(prob, 2)})
        
    results.sort(key=lambda x: x["prob"], reverse=True)
    return results

# ============================================================
# 4. Streamlit UI & 実行処理
# ============================================================
st.title("🚤 v20 波乱特化AI (スジ連動・キューマスター対応)")

st.sidebar.header("⚙️ 予想設定")
target_date = st.sidebar.date_input("予想する日付", datetime.now(JST))
date_str = target_date.strftime("%Y%m%d")

venue_name = st.sidebar.selectbox("競艇場", ["全場一括処理"] + list(JCD_NAME.values()))
target_jcds = list(JCD_NAME.keys()) if venue_name == "全場一括処理" else [[k for k, v in JCD_NAME.items() if v == venue_name][0]]

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 買い目フィルター")
top_n = st.sidebar.number_input("上位何点まで買うか？ (推奨: 1〜4)", min_value=1, max_value=20, value=1)
min_prob = st.sidebar.slider("勝率の足切り下限(%) (推奨: 11%)", min_value=0.0, max_value=50.0, value=11.0, step=1.0)
max_prob = st.sidebar.slider("勝率の足切り上限(%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
bet_amount = st.sidebar.number_input("1点あたりの購入金額 (円)", min_value=100, max_value=10000, value=100, step=100)

def process_single_race(jcd, rno, dstr):
    res = fetch_kyotei24_data(jcd, rno, dstr)
    if not res:
        return {"日付": dstr, "場": JCD_NAME[jcd], "R": f"{rno}R", "買い目": "データなし", "結果": "-", "結果確率": "-", "的中": "-", "払戻金": 0, "点数": 0}
        
    racers, actual_result, payoff, has_result = res
    probs = calculate_v20_probs(jcd, racers)
    
    top_n_bets = probs[:int(top_n)]
    buy_bets = [b for b in top_n_bets if min_prob <= b["prob"] <= max_prob]
    
    kaime_str = "見"
    count = 0
    hit_mark = "❌"
    result_prob_str = "-"
    actual_payoff = 0
    
    if buy_bets:
        kaime_str = ", ".join([f"{b['bet']}({b['prob']}%)" for b in buy_bets])
        count = len(buy_bets)
        
    if has_result and actual_result:
        # 結果の確率を取得
        res_prob = next((b["prob"] for b in probs if b["bet"] == actual_result), 0.0)
        result_prob_str = f"{actual_result}({res_prob}%)"
        
        # 的中判定
        if buy_bets:
            for b in buy_bets:
                if b["bet"] == actual_result:
                    hit_mark = "🎯"
                    actual_payoff = payoff
                    break
        
    return {
        "日付": dstr, "場": JCD_NAME[jcd], "R": f"{rno}R", 
        "買い目": kaime_str, "結果": actual_result if has_result else "結果待ち", 
        "結果確率": result_prob_str if has_result else "⏳", 
        "的中": hit_mark if has_result else "-", 
        "払戻金": actual_payoff, "点数": count
    }

if st.button("🚀 予想 / バックテストを実行"):
    if not model_12 or not model_13:
        st.error("モデルが読み込まれていないため実行できません。")
        st.stop()

    results_disp = []
    tasks = [(j, r) for j in target_jcds for r in range(1, 13)]
    
    with st.spinner(f"⚡ データ取得＆予測中 (全 {len(tasks)} レース)..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_task = {executor.submit(process_single_race, j, r, date_str): (j, r) for j, r in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                results_disp.append(future.result())
                
    # 表示順を整える
    df = pd.DataFrame(results_disp)
    df['場_id'] = df['場'].map({v: k for k, v in JCD_NAME.items()})
    df['R_num'] = df['R'].str.replace('R', '').astype(int)
    df = df.sort_values(by=['場_id', 'R_num']).drop(columns=['場_id', 'R_num']).reset_index(drop=True)
    
    # 画面上部：実績サマリー
    total_bets = df["点数"].sum()
    total_cost = total_bets * bet_amount
    total_payoff = df["払戻金"].sum() * (bet_amount / 100) # 100円単位で計算
    roi = (total_payoff / total_cost * 100) if total_cost > 0 else 0
    
    st.success(f"✅ 処理完了！ 購入点数: {total_bets}点 (投資: {total_cost:,.0f}円) / 回収: {total_payoff:,.0f}円 / 回収率: {roi:.1f}%")

    # 結果表示用データフレーム
    disp_cols = ["日付", "場", "R", "買い目", "結果", "結果確率", "的中", "払戻金"]
    st.dataframe(df[disp_cols], use_container_width=True)

    # ============================================================
    # 5. 全自動購入用データ (キューマスター専用) 生成
    # ============================================================
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
    
    if auto_bet_queue:
        json_string = json.dumps(auto_bet_queue, ensure_ascii=False, indent=4)
        st.markdown("---")
        st.subheader("🤖 全自動購入用データ (キューマスター専用)")
        st.caption("右上のコピーボタンを押して、キューマスターに貼り付けてください。")
        st.code(json_string, language="json")
    else:
        st.info("条件に合致する買い目がありませんでした。設定（閾値など）を見直してください。")
