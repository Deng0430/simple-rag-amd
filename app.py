"""A small, sourced AUP onboarding assistant with a Streamlit interface."""

import os
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import streamlit as st

from rag import DEFAULT_DATA, DEFAULT_MODEL, Generator, Retriever, environment, export_csv, load_data, run_query

OFFICIAL_GUIDE = "https://amdresearch.github.io/aup-learning-cloud/user-guide/platform-basics.html"
REPOSITORY = "https://github.com/Deng0430/simple-rag-amd"
SAMPLES = {
    "aup-q01": ("找不到 Python 命令", ":material/terminal:"),
    "aup-q02": ("虚拟环境创建失败", ":material/build:"),
    "aup-q03": ("网页启动了却打不开", ":material/language:"),
    "aup-q04": ("会话到期，怎么保存", ":material/save:"),
    "aup-q05": ("AMD 为何用 cuda 接口", ":material/memory:"),
    "aup-q06": ("该选 CPU 还是 GPU", ":material/dns:"),
}


def icon(name, size=20):
    symbols = {
        "terminal": "terminal", "book": "menu_book", "cloud": "cloud_upload",
        "globe": "language", "save": "save", "search": "search",
    }
    # Reuse Streamlit's bundled icon font; st.html strips inline SVGs.
    return (f'<span class="aup-icon" style="font-size:{size}px" '
            f'aria-hidden="true" translate="no">{symbols[name]}</span>')


st.set_page_config(page_title="AUP 开发助手", page_icon="☁", layout="wide")
st.html(f"<style>{(Path(__file__).parent / 'assets/style.css').read_text(encoding='utf-8')}</style>")


@st.cache_resource
def resources(data_revision):
    # A data update must invalidate the cached index on the next page interaction.
    data = load_data()
    return data, Retriever(data["passages"], **data.get("retriever", {}))


@st.cache_resource(max_entries=1)
def model_resource(model_id):
    return Generator(model_id)


@st.cache_data(ttl=60)
def runtime_info():
    return environment()


def reset_results():
    st.session_state.pop("results", None)


def use_example(question):
    st.session_state["question"] = question
    reset_results()


def show_passage(passage):
    st.html(f'<div class="passage-scope">适用范围：{escape(passage["scope"])}</div>')
    st.markdown(passage["text"])
    st.html('<div class="source-label">资料来源</div>')
    for source in passage["sources"]:
        st.html(
            '<div class="source-item">'
            f'<a href="{escape(source["url"], quote=True)}" target="_blank" rel="noopener noreferrer">'
            f'{escape(source["title"])}</a>'
            f'<small>{escape(source["kind"])} / 核对于 {escape(source["checked_on"])}</small></div>'
        )


def show_examples(questions):
    with st.container(key="sample-questions"):
        for start in range(0, len(questions), 2):
            for column, item in zip(st.columns(2), questions[start:start + 2]):
                label, symbol = SAMPLES.get(item["id"], (item["question"], ":material/help:"))
                column.button(label, icon=symbol, key=f'sample_{item["id"]}',
                              use_container_width=True, on_click=use_example,
                              args=(item["question"],), help=item["question"])


def show_result(result, *, paired=False):
    if paired:
        st.markdown(f"### 最多检索 {result['top_k']} 条")
    if result["notice"]:
        st.warning(result["notice"])
        st.link_button("查阅 AUP 官方指南", OFFICIAL_GUIDE, icon=":material/open_in_new:")
    if result["answer"] is not None:
        with st.container(key=f"answer-{result['top_k'] if paired else 'single'}"):
            st.markdown("### 操作建议")
            st.markdown(result["answer"])
            st.caption("建议依据下方指南整理，可按 [P1] 等编号核对出处。")
    if result["passages"]:
        st.markdown(f"### 相关指南（{result['retrieved_count']} 条）")
        if result["mode"] == "retrieval_preview":
            st.caption("以下为指南原文。当前使用检索预览，尚未生成模型回答。")
        for number, passage in enumerate(result["passages"], 1):
            with st.expander(f"[P{number}] {passage['title']}",
                             expanded=number == 1 and result["mode"] == "retrieval_preview"):
                show_passage(passage)
    with st.expander("运行记录与核对", icon=":material/analytics:"):
        a, b, c = st.columns(3)
        a.metric("总耗时", f"{result['total_seconds']:.3f} s")
        b.metric("输入 token", result["input_tokens"] if result["input_tokens"] is not None else "—")
        c.metric("输出 token", result["output_tokens"] if result["output_tokens"] is not None else "—")
        if result.get("reference_answers"):
            st.write("核对要点：" + result["reference_answers"])
            st.caption("参考指南已检索到" if result["reference_passage_retrieved"] else "未检索到参考指南")
        st.caption("计时不含首次下载、加载、预热和网页渲染。仅检索时不生成答案、不计模型 token。")


data, retriever = resources(DEFAULT_DATA.stat().st_mtime_ns)
env = runtime_info()
gpu_available = bool(env.get("gpu"))
st.session_state.setdefault("question", "")

with st.sidebar:
    st.html(
        f'<div class="aup-brand"><div class="aup-brand-mark">{icon("cloud", 25)}</div>'
        '<div><div class="aup-brand-title">AUP 开发助手</div>'
        '<div class="aup-brand-subtitle">云端入门，有据可查</div></div></div>'
    )
    st.html('<div class="sidebar-label">回答方式</div>')
    mode = st.radio("回答方式", ["仅检索预览", "模型问答"], index=1 if gpu_available else 0,
                    key="mode", label_visibility="collapsed", on_change=reset_results)
    st.caption("检索预览直接查阅指南；模型问答会根据指南整理操作建议。")
    status_title = "GPU 已就绪" if gpu_available else "资料检索已就绪"
    status_detail = escape(env["gpu"]) if gpu_available else "当前环境未检测到 GPU，可使用检索预览。"
    st.html(f'<div class="runtime-status"><strong><span class="status-dot"></span>{status_title}</strong>'
            f'<small>{status_detail}</small></div>')
    if mode == "模型问答" and not gpu_available:
        st.info("模型问答需要 GPU。请在 AUP 的 GPU 服务器启动本项目，或切回检索预览。")
    st.html('<div class="sidebar-label">可以查阅的内容</div>')
    st.html("".join(
        f'<div class="sidebar-topic">{icon(symbol, 18)}<span>{label}</span></div>'
        for symbol, label in [("terminal", "Python 与 GPU 环境"), ("globe", "启动网页与端口访问"), ("save", "保存文件与管理会话")]
    ))
    with st.expander("模型与运行环境", icon=":material/settings:"):
        model_id = st.text_input("生成模型", value=os.getenv("MODEL_ID", DEFAULT_MODEL),
                                 disabled=mode == "仅检索预览", on_change=reset_results)
        st.json(env)
    with st.expander("演示参数", icon=":material/tune:"):
        k = st.select_slider("最多检索几条指南", options=[3, 6], value=3)
        paired = st.checkbox("同时比较最多 3 条与 6 条指南", value=False)
        st.caption("只选足够相关的指南，实际数量可能更少。对比用于观察回答和运行开销。")
    st.html(f'<div class="sidebar-footnote">收录 {len(data["passages"])} 条指南<br>资料核对日期：{escape(data["updated"])}</div>')
    st.link_button("AUP 官方使用指南", OFFICIAL_GUIDE, icon=":material/open_in_new:", use_container_width=True)

badge = "检索预览" if mode == "仅检索预览" else ("GPU 模型问答" if gpu_available else "模型问答 · 需要 GPU")
st.html(f'<div class="page-meta"><span class="page-meta-label">AMD AUP Learning Cloud 入门指南</span>'
        f'<span class="mode-badge">{icon("search" if mode == "仅检索预览" else "cloud", 15)}{badge}</span></div>')
st.html('<h1>云端开发，下一步怎么做？</h1><p class="page-intro">从配置 Python 到打开网页，描述你遇到的问题，找到操作指南和回答依据。</p>')

ask_tab, guides_tab = st.tabs(["问题解答", f"指南库 · {len(data['passages'])}"])
with ask_tab:
    with st.container(key="question-panel"):
        st.html(f'<div class="question-title">{icon("terminal", 22)}<h2>你遇到了什么问题？</h2></div>')
        with st.form("question-form", border=False):
            question = st.text_area("你的问题", key="question", height=130, label_visibility="collapsed",
                                    placeholder="粘贴报错，或描述你卡住的步骤。\n例如：Streamlit 已经启动，为什么浏览器打不开？")
            hint_column, action_column = st.columns([1.5, 1], vertical_alignment="center")
            with hint_column:
                hint = "直接查阅原文，附来源链接" if mode == "仅检索预览" else "首次问答需要加载模型，请稍候"
                st.html(f'<div class="question-hint">{hint}</div>')
            with action_column:
                single = st.form_submit_button(
                    "获取操作建议" if mode == "模型问答" else "查找相关指南",
                    type="primary", icon=":material/search:", use_container_width=True,
                    disabled=mode == "模型问答" and not gpu_available,
                )

    if single:
        reset_results()
        if not question.strip():
            st.warning("请先输入问题，也可以点击下方常见问题填入。")
        elif mode == "模型问答" and not model_id.strip():
            st.warning("请在侧栏的“模型与运行环境”中填写模型名称。")
        else:
            try:
                generator = None
                if mode == "模型问答" and retriever.search(question, k):
                    with st.spinner("正在加载并预热模型，首次运行需要一些时间…"):
                        generator = model_resource(model_id.strip())
                with st.spinner("正在查阅指南…"):
                    results = []
                    selected = next((q for q in data["questions"] if q["question"] == question.strip()), None)
                    for top_k in ([3, 6] if paired else [k]):
                        result = run_query(retriever, question, top_k, generator,
                                           profile=data["prompt_profile"], max_new_tokens=data["max_new_tokens"],
                                           knowledge_base=data["id"])
                        result["recorded_at"] = datetime.now(timezone.utc).isoformat()
                        if selected:
                            result["question_id"] = selected["id"]
                            result["reference_answers"] = " | ".join(selected["answers"])
                            result["reference_passage_retrieved"] = any(
                                p["id"] == selected["passage_id"] for p in result["passages"]
                            )
                        results.append(result)
                    st.session_state["results"] = results
            except Exception as exc:
                st.error("本次问答未完成。请检查运行环境后重试，或切换到检索预览。")
                with st.expander("查看错误详情", expanded=True):
                    st.code(str(exc), language="text")

    results = st.session_state.get("results", [])
    home_examples = [q for q in data["questions"] if q["id"] in SAMPLES]
    if not results:
        st.html('<div class="suggestions-label">也可以从常见问题开始，点击填入</div>')
        show_examples(home_examples)
        st.html(f'<div class="empty-help">{icon("book", 20)}<span>先查指南，再看出处。<br>每条指南都附适用范围，可随时在“指南库”中阅读。</span></div>')
    else:
        with st.expander("换个常见问题", icon=":material/chat_bubble_outline:"):
            show_examples(home_examples)
        with st.container(key="results"):
            st.html('<div class="result-heading"><h2>查阅结果</h2></div>')
            st.caption(f"本次问题：{results[0]['question']}")
            columns = st.columns(len(results))
            for column, result in zip(columns, results):
                with column:
                    show_result(result, paired=len(results) > 1)
            st.download_button("下载本次结果 CSV", export_csv(results), file_name="aup_demo_results.csv",
                               mime="text/csv", icon=":material/download:")

with guides_tab:
    st.html('<div class="library-heading"><h2>按主题查阅指南</h2></div>')
    st.caption("根据官方文档与本项目部署记录整理，可直接阅读，也可通过提问查找。")
    guide_search = st.text_input("搜索指南", placeholder="输入标题、报错或关键词，例如：端口占用、git pull、模型缓存", key="guide-search")
    categories = list(dict.fromkeys(p["category"] for p in data["passages"]))
    category = st.pills("筛选指南主题", ["全部"] + categories, default="全部", key="guide-category",
                        label_visibility="collapsed") or "全部"
    terms = guide_search.casefold().split()
    passages = [p for p in data["passages"]
                if (category == "全部" or p["category"] == category)
                and all(term in (p["title"] + " " + p["keywords"] + " " + p["text"]).casefold()
                        for term in terms)]
    st.html(f'<div class="library-count">{escape(category)} / {len(passages)} 条指南</div>')
    if not passages:
        st.info("没有符合条件的指南。试试其他关键词，或将主题切换为“全部”。")
    with st.container(key="guide-index"):
        for passage in passages:
            with st.expander(passage["title"]):
                st.caption(passage["category"])
                show_passage(passage)

st.html('<footer class="page-footer"><span>课程演示项目，非 AUP 官方客服。资源与账号状态以平台为准。</span>'
        f'<a href="{REPOSITORY}" target="_blank" rel="noopener noreferrer">GitHub 开源项目</a></footer>')
