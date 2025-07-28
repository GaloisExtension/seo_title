"""
Streamlit + LangChain based keyword & article-volume research tool
------------------------------------------------------------------

目的:
- LLMにより大量の検索キーワードを自動生成
- Google / Note / Qiita / Zenn などから記事件数やタイトル/タグを取得
- LLMで勝ちパターン抽出・ギャップ分析

この1ファイル版は **とりあえず動く最小構成(MVP)** です。
本番向けには /modules ディレクトリへ分割推奨。

必要環境変数:
- OPENAI_API_KEY or AZURE_OPENAI_API_KEY（使用モデルに合わせて）
- SERPAPI_KEY（任意: Google検索件数用。未設定ならDuckDuckGoにフォールバック）

実装メモ:
- Qiitaは公式APIを叩きます（Rate Limit: 60req/min）。
- Zenn/Note は公式APIが弱いためHTMLスクレイピング(robots.txt確認要)。必要に応じて手動CSVアップロードのUIも用意。
- Google Trends は pytrends を使用(任意)。
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# -----------------------------
# ----- Auth helper -----------
# -----------------------------


# 認証情報を secrets.toml から取得（無ければデフォルト）
AUTH_USERNAME = None
AUTH_PASSWORD = None
try:
    AUTH_USERNAME = st.secrets.get("AUTH_USERNAME")  # type: ignore[attr-defined]
    AUTH_PASSWORD = st.secrets.get("AUTH_PASSWORD")  # type: ignore[attr-defined]
except Exception:
    pass

if not AUTH_USERNAME:
    AUTH_USERNAME = "mellon"
if not AUTH_PASSWORD:
    AUTH_PASSWORD = "pw"

# シンプルに 1 ユーザー想定


def login_screen() -> bool:
    """シンプルなログイン画面を表示し、認証済なら True を返す"""
    if st.session_state.get("authenticated"):
        return True

    st.title("🔐 ログイン")
    st.write("ユーザー名とパスワードを入力してください。")

    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == AUTH_USERNAME and (AUTH_PASSWORD == "*" or pw == AUTH_PASSWORD):
            st.session_state["authenticated"] = True
            st.success("Login success!")
            # rerun in a version-agnostic way
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
            elif hasattr(st, "rerun"):
                st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# -----------------------------
# ---------- Config -----------
# -----------------------------
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = 0.2
# ----- APIキー設定 -----
# Streamlit Cloud 等では st.secrets、ローカルでは環境変数から取得する
# secrets.toml が存在しない環境では st.secrets へのアクセス自体が例外になるため try で包む
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # type: ignore[index]
    except Exception:
        OPENAI_API_KEY = ""

# Google Custom Search JSON API キー / CX
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
if not GOOGLE_API_KEY or not GOOGLE_CX:
    try:
        GOOGLE_API_KEY = GOOGLE_API_KEY or st.secrets["GOOGLE_API_KEY"]  # type: ignore[index]
        GOOGLE_CX = GOOGLE_CX or st.secrets["GOOGLE_CX"]  # type: ignore[index]
    except Exception:
        pass
# APIキーが見つからなければサイドバーで警告を表示して以降の処理が落ちないようにする
if not OPENAI_API_KEY:
    st.sidebar.error(
        "OPENAI_API_KEY が設定されていません。\n"
        "ターミナルで `export OPENAI_API_KEY=YOUR_KEY` を実行するか、"
        ".streamlit/secrets.toml に OPENAI_API_KEY を追加してください。"
    )
else:
    # langchain_openai は環境変数を参照するため保証しておく
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.set_page_config(page_title="TS-AI Keyword Research", layout="wide")

# -----------------------------
# ----- Data Models -----------
# -----------------------------
class GeneratedKeyword(BaseModel):
    keyword: str = Field(..., description="検索に用いるキーワード")
    intent: str = Field(..., description="このキーワードの検索意図の説明")
    variants: List[str] = Field(default_factory=list, description="類似/派生キーワード")

class KeywordSet(BaseModel):
    theme: str
    industry: str
    tech_terms: List[str]
    base_keywords: List[GeneratedKeyword]

# PatternInsight: suggestions が dict で返ることがあるため Any で許容
class PatternInsight(BaseModel):
    title_patterns: List[str]
    hashtag_patterns: List[str]
    gaps: List[str]
    suggestions: List[Any]

# -----------------------------
# ----- LLM Helpers -----------
# -----------------------------
# サイドバーからの入力にも対応するため api_key を引数に追加
@st.cache_resource(show_spinner=False)
def get_llm(model: str = DEFAULT_MODEL, api_key: str = "") -> ChatOpenAI:
    # api_key が渡されていれば優先。なければ環境変数・st.secrets を使用
    key = api_key or os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(model=model, temperature=TEMPERATURE, openai_api_key=key)


def build_keyword_prompt(theme: str, industries: List[str], tech_terms: List[str]) -> ChatPromptTemplate:
    template = (
        """あなたはSEO/コンテンツマーケの専門家です。以下の要件で検索キーワード候補を作ってください。\n\n"
        "主題: {theme}\n"
        "業界: {industries}\n"
        "関連技術: {tech_terms}\n\n"
        "出力要件:\n"
        "- 各業界×技術×主題の掛け合わせを意識して多様に\n"
        "- 日本語キーワードを中心に、必要なら英語も\n"
        "- 1行に1キーワード。intent(検索意図)とvariants(派生形)も付ける\n"
        "- JSONで返す。" 
        """
    )
    return ChatPromptTemplate.from_template(template)


keyword_parser = PydanticOutputParser(pydantic_object=List[GeneratedKeyword])


def generate_keywords(llm: ChatOpenAI, theme: str, industries: List[str], tech_terms: List[str]) -> List[GeneratedKeyword]:
    prompt = build_keyword_prompt(theme, industries, tech_terms)
    messages = prompt.format_messages(theme=theme, industries=industries, tech_terms=tech_terms)
    # Force JSON mode via tool like structured output or by asking
    resp = llm.invoke(messages + [
        {"role": "system", "content": "必ず有効なJSON配列のみを出力。余計な文章禁止。"}
    ])
    text = resp.content.strip()
    try:
        data = json.loads(text)
        return [GeneratedKeyword(**d) for d in data]
    except Exception:
        # fallback: try to extract JSON
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            data = json.loads(m.group(0))
            return [GeneratedKeyword(**d) for d in data]
        st.error("LLMの出力JSON解析に失敗しました。プロンプトを調整してください。\n\n出力:\n" + text)
        return []


pattern_parser = PydanticOutputParser(pydantic_object=PatternInsight)

def analyze_patterns(llm: ChatOpenAI, titles: List[str], tags: List[str], counts: Dict[str, int]) -> PatternInsight:
    template = (
        """あなたはコンテンツマーケのストラテジストです。以下のデータを分析し、Noteでの勝ちパターンを抽出してください。\n\n"
        "【タイトル一覧】\n{titles}\n\n"
        "【ハッシュタグ一覧】\n{tags}\n\n"
        "【キーワード別記事数】\n{counts}\n\n"
        "出力要件:\n"
        "- 効果的なタイトルの型（title_patterns）\n"
        "- 強いハッシュタグパターン（hashtag_patterns）\n"
        "- 記事が少ないが需要がありそうなギャップテーマ（gaps）\n"
        "- 具体的な記事案の提案（suggestions）\n"
        "JSONで返す。"""
    )
    prompt = ChatPromptTemplate.from_template(template)
    messages = prompt.format_messages(titles=titles, tags=tags, counts=counts)
    resp = llm.invoke(messages + [
        {"role": "system", "content": "必ず有効なJSONのみ。余計な文章は禁止。"}
    ])
    text = resp.content.strip()
    try:
        data = json.loads(text)
        return PatternInsight(**data)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            data = json.loads(m.group(0))
            return PatternInsight(**data)
        st.error("JSON解析失敗: " + text)
        raise

# -----------------------------
# ----- Scrapers --------------
# -----------------------------

# Minimalistic wrappers. In production, handle retries, headers, error codes, robots.txt.

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def google_result_count(
    query: str,
    serpapi_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    google_cx: Optional[str] = None,
) -> int:
    """検索結果件数を返す。

    優先順:
    1. SerpAPI (安定 / 高速)
    2. Google Custom Search JSON API
    3. HTML スクレイピング (簡易)
    """
    if serpapi_key:
        url = "https://serpapi.com/search.json"
        params = {"q": query, "engine": "google", "api_key": serpapi_key}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return int(data.get("search_information", {}).get("total_results", 0))

    if google_api_key and google_cx:
        url = "https://customsearch.googleapis.com/customsearch/v1"
        params = {"key": google_api_key, "cx": google_cx, "q": query}
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:  # quota exceeded
            return -2
        r.raise_for_status()
        data = r.json()
        return int(data.get("searchInformation", {}).get("totalResults", 0))

    # fallback: scrape (rough)
    headers = {"User-Agent": USER_AGENT}
    r = requests.get("https://www.google.com/search", params={"q": query}, headers=headers, timeout=30)
    if r.status_code != 200:
        return -1
    m = re.search(r"約?([0-9,]+)件", r.text)
    if m:
        return int(m.group(1).replace(",", ""))
    return -1


def qiita_search_count(query: str) -> Tuple[int, List[Dict[str, Any]]]:
    url = "https://qiita.com/api/v2/items"
    params = {"query": query, "per_page": 20, "page": 1}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return -1, []
    items = r.json()
    return len(items), items

# -----------------------------
# ----- Streamlit UI ----------
# -----------------------------

def sidebar_inputs() -> Dict[str, Any]:
    st.sidebar.header("基本設定")
    theme = st.sidebar.text_input("主題", value="時系列分析")
    inds = st.sidebar.text_input("対象業界(カンマ区切り)", value="製造業,小売業").split(",")
    inds = [x.strip() for x in inds if x.strip()]
    tech = st.sidebar.text_input("技術キーワード(カンマ区切り)", value="在庫最適化,発注自動化").split(",")
    tech = [x.strip() for x in tech if x.strip()]

    st.sidebar.header("APIキー設定")
    model = st.sidebar.text_input("LLMモデル", value=DEFAULT_MODEL)
    openai_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", placeholder="sk-...")
    serpapi_key = st.sidebar.text_input("SERPAPI_KEY (任意)", type="password")
    google_api_key = st.sidebar.text_input("GOOGLE_API_KEY (任意)", type="password")
    google_cx = st.sidebar.text_input("GOOGLE_CX 検索エンジンID (任意)")

    return {
        "theme": theme,
        "industries": inds,
        "tech": tech,
        "model": model,
        "openai_api_key": openai_key,
        "serpapi_key": serpapi_key,
        "google_api_key": google_api_key,
        "google_cx": google_cx,
    }


def main():
    login_screen() # ログイン画面を表示

    cfg = sidebar_inputs()
    # ユーザーがAPIキーを入力した場合は環境変数にも設定しておく（再読込や他ライブラリが参照する可能性に備える）
    if cfg.get("openai_api_key"):
        os.environ["OPENAI_API_KEY"] = cfg["openai_api_key"]

    llm = get_llm(cfg["model"], cfg.get("openai_api_key", ""))

    st.title("SEOキーワード生成")

    # 1) LLMでキーワード生成
    st.header("1. キーワード自動生成")
    if st.button("キーワード生成する", type="primary"):
        with st.spinner("LLMがキーワードを生成中..."):
            kws = generate_keywords(llm, cfg["theme"], cfg["industries"], cfg["tech"])
        if kws:
            df_kw = pd.DataFrame([k.model_dump() for k in kws])
            st.session_state["keywords_df"] = df_kw
            st.success(f"{len(df_kw)}件のキーワード生成")
    if "keywords_df" in st.session_state:
        st.download_button(
            "キーワードCSVをダウンロード",
            st.session_state["keywords_df"].to_csv(index=False).encode("utf-8"),
            file_name="keywords.csv",
            mime="text/csv",
        )
        # 表示を常に維持
        st.dataframe(st.session_state["keywords_df"], use_container_width=True)

    # 2) 各媒体の検索＆収集
    st.header("2. 媒体別 検索・件数集計")
    if "keywords_df" not in st.session_state:
        st.info("まずキーワードを生成してください。")
    else:
        df_kw = st.session_state["keywords_df"]
        selected_kws = st.multiselect("調査するキーワードを選択", df_kw["keyword"].tolist(), df_kw["keyword"].tolist()[:10])
        max_titles = st.number_input("各媒体から取得するタイトル上限", 5, 100, 20)
        if st.button("検索を実行"):
            serp = cfg["serpapi_key"] or None
            g_counts = {}
            qiita_rows = []
            # Note / Zenn は取得しない

            progress = st.progress(0.0)
            total = len(selected_kws)
            for i, q in enumerate(selected_kws):
                g_counts[q] = google_result_count(
                    q,
                    serpapi_key=serp,
                    google_api_key=cfg.get("google_api_key") or GOOGLE_API_KEY,
                    google_cx=cfg.get("google_cx") or GOOGLE_CX,
                )
                qiita_c, qiita_items = qiita_search_count(q)
                qiita_rows.extend([{**item, "_kw": q} for item in qiita_items])
                # Note / Zenn は取得しない
                progress.progress((i + 1) / total)
                time.sleep(0.3)

            df_google = pd.DataFrame({"keyword": list(g_counts.keys()), "google_results": list(g_counts.values())})
            df_qiita = pd.DataFrame(qiita_rows)
            # Note / Zenn は生成しない

            st.session_state.update({
                "df_google": df_google,
                "df_qiita": df_qiita,
                # Note / Zenn は保存しない
            })
            st.success("検索完了！")

    if "df_google" in st.session_state:
        st.subheader("Google検索件数")
        df_google_disp = st.session_state["df_google"].sort_values("google_results", ascending=False)
        st.dataframe(df_google_disp.reset_index(drop=True), use_container_width=True)
    if "df_qiita" in st.session_state:
        st.subheader("Qiita記事一覧(いいね順)")
        df_qiita_raw = st.session_state["df_qiita"].copy()
        if "tags" in df_qiita_raw.columns:
            df_qiita_raw["tags"] = df_qiita_raw["tags"].apply(
                lambda l: ",".join(t["name"] for t in l) if isinstance(l, list) else ""
            )
        else:
            df_qiita_raw["tags"] = ""

        if "likes_count" in df_qiita_raw.columns:
            df_qiita_raw = df_qiita_raw.sort_values("likes_count", ascending=False)
        qiita_display = df_qiita_raw.reset_index(drop=True)
        display_cols = [c for c in ["_kw", "title", "likes_count", "tags", "url"] if c in qiita_display.columns]
        st.dataframe(qiita_display[display_cols].fillna(""), use_container_width=True)
    # Note / Zenn 表示は行わない

    # 3) LLMで勝ちパターン分析
    st.header("3. LLMで勝ちパターン分析")
    if all(k in st.session_state for k in ["df_qiita", "df_google"]):
        if "title" not in st.session_state["df_qiita"].columns or st.session_state["df_qiita"].empty:
            st.warning("Qiita からタイトルを取得できませんでした。キーワードや API レートを確認してください。")
            titles = []
        else:
            titles = st.session_state["df_qiita"]["title"].tolist()
        # Qiita tags
        tag_list = []
        if "df_qiita" in st.session_state and not st.session_state["df_qiita"].empty and "tags" in st.session_state["df_qiita"].columns:
            for tags in st.session_state["df_qiita"]["tags"]:
                if isinstance(tags, list):
                    tag_list.extend([t.get("name", "") for t in tags])
        if titles and st.button("勝ちパターン抽出"):
            with st.spinner("LLMが分析中..."):
                insight = analyze_patterns(llm, titles, tag_list, st.session_state["df_google"].set_index("keyword")["google_results"].to_dict())
            st.json(insight.model_dump())
    else:
        st.info("検索結果（Qiita と Google）が揃ってから分析してください。")

    if "df_trend" in st.session_state:
        st.subheader("Google Trends 人気度 (0-100)")
        st.dataframe(st.session_state["df_trend"].sort_values("trend_score", ascending=False), use_container_width=True)

    st.markdown("---")
    st.caption("© 2025 Keyword Research MVP")


if __name__ == "__main__":
    main()
