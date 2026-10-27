"""A small, sourced AUP onboarding assistant with a Streamlit interface."""

import os
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import streamlit as st

from rag import (DEFAULT_MODEL, DEFAULT_LOCAL_MODEL, DEFAULT_OLLAMA_URL,
                 create_generator, environment, export_csv, ollama_status, run_query)
from knowledge import CORPUS, load_knowledge, signature
from retrievers import LABELS, create_retriever

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
    return load_knowledge()


@st.cache_resource(max_entries=6)
def retrieval_resource(data_revision, method, min_score):
    data = resources(data_revision)
    return create_retriever(data["passages"], method, min_score=min_score)


@st.cache_resource(max_entries=1)
def model_resource(backend, model_id, base_url, num_ctx):
    return create_generator(backend, model_id, base_url=base_url, num_ctx=num_ctx)


@st.cache_data(ttl=10)
def local_model_status(base_url):
    return ollama_status(base_url)


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
        st.markdown(f"### {LABELS[result['retriever']].split(' · ')[0]}")
    st.caption(f"{LABELS[result['retriever']]} / Top-k {result['top_k']} / 检索 {result['retrieval_seconds']:.3f} s")
    if result["notice"]:
        st.warning(result["notice"])
        st.link_button("查阅 AUP 官方指南", OFFICIAL_GUIDE, icon=":material/open_in_new:")
    if result["answer"] is not None:
        with st.container(key=f"answer-{result['top_k'] if paired else 'single'}"):
            st.markdown("### 操作建议")
            st.markdown(result["answer"])
            st.caption("建议依据下方指南整理，可按 [P1] 等编号核对出处。")
    if result["passages"]:
        st.markdown(f"### 相关资料（{result['retrieved_count']} 段）")
        if result["mode"] == "retrieval_preview":
            st.caption("以下是检索到的资料片段。分数不是正确率，不同检索器的数值不可直接比较。")
        for number, passage in enumerate(result["passages"], 1):
            st.caption(f"排名 {number} / {result['score_kind']} {passage['score']:.4f}")
            with st.expander(f"[P{number}] {passage['title']}",
                             expanded=number == 1 and result["mode"] == "retrieval_preview" and not paired):
                show_passage(passage)
    with st.expander("运行记录与核对", icon=":material/analytics:"):
        a, b, c = st.columns(3)
        a.metric("总耗时", f"{result['total_seconds']:.3f} s")
        b.metric("输入 token", result["input_tokens"] if result["input_tokens"] is not None else "—")
        c.metric("输出 token", result["output_tokens"] if result["output_tokens"] is not None else "—")
        st.caption(f"分数门槛：{result['min_score']}；索引准备耗时：{result.get('index_seconds', 0) or 0:.3f} s（首次准备可能包括下载与编码）")
        if result.get("retriever_model"):
            st.caption("嵌入模型：" + result["retriever_model"] + " / CPU ONNX")
            st.caption(f"向量索引：{result['indexed_passages']} × {result['index_dimensions']}；"
                       + ("复用磁盘缓存" if result["index_cache_hit"] else "本次完成文档编码并保存缓存"))
        if result.get("reference_answers"):
            st.write("核对要点：" + result["reference_answers"])
            st.caption("参考指南已检索到" if result["reference_passage_retrieved"] else "未检索到参考指南")
        if result.get("backend") == "ollama":
            st.caption(f"生成模型：{result['model']} / Ollama / {result.get('model_quantization', '未知量化')}")
            st.caption("请求计时包含本次服务调用；首次预热单列。模型重新加载时，加载耗时会包含在请求内。")
            if result.get("load_seconds") is not None:
                st.caption(f"本次服务报告加载耗时：{result['load_seconds']:.3f} s；结束原因：{result.get('done_reason', '未知')}")
        else:
            st.caption("计时不含首次下载、加载、预热和网页渲染。仅检索时不生成答案、不计模型 token。")


data_revision = signature()
data = resources(data_revision)
env = runtime_info()
gpu_available = bool(env.get("gpu"))
st.session_state.setdefault("question", "")

with st.sidebar:
    st.html(
        f'<div class="aup-brand"><div class="aup-brand-mark">{icon("cloud", 25)}</div>'
        '<div><div class="aup-brand-title">AUP 开发助手</div>'
        '<div class="aup-brand-subtitle">云端入门，有据可查</div></div></div>'
    )
    with st.expander("模型与运行环境", icon=":material/settings:"):
        backends = {"transformers": "AUP / Transformers", "ollama": "本地 / Ollama"}
        backend = st.selectbox("模型运行方式", list(backends), format_func=backends.get,
                               index=1 if os.getenv("RAG_BACKEND") == "ollama" else 0,
                               key="generation-backend", on_change=reset_results)
        default_model = DEFAULT_LOCAL_MODEL if backend == "ollama" else DEFAULT_MODEL
        if backend == os.getenv("RAG_BACKEND", "transformers"):
            default_model = os.getenv("MODEL_ID", default_model)
        model_id = st.text_input("生成模型", value=default_model,
                                 key="model-" + backend, on_change=reset_results)
        base_url = DEFAULT_OLLAMA_URL
        num_ctx = 8192
        local_status = None
        if backend == "ollama":
            base_url = st.text_input("Ollama 服务地址", value=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
                                     on_change=reset_results)
            num_ctx = int(st.number_input("上下文窗口（token）", min_value=512, max_value=32768,
                                          value=8192, step=512, on_change=reset_results))
            local_status = local_model_status(base_url)
            if st.button("刷新模型状态"):
                local_model_status.clear()
                st.rerun()
            st.caption("使用 Ollama 已安装的模型；首次问答会加载并预热。")
            st.json(local_status)
        st.caption("以下是网页 Python 环境；Ollama 在独立进程中运行。" if backend == "ollama" else "当前 Python 环境")
        st.json(env)
    if backend == "ollama":
        model_names = {item["name"] for item in local_status.get("models", [])}
        generation_available = local_status["available"] and model_id.strip() in model_names
        status_title = "本地模型已就绪" if generation_available else "本地模型待连接"
        status_detail = (model_id.strip() + " / Ollama") if generation_available else (
            local_status.get("error") or "Ollama 中未找到该模型，请检查名称。")
    else:
        generation_available = gpu_available
        status_title = "GPU 已就绪" if gpu_available else "资料检索已就绪"
        status_detail = env["gpu"] if gpu_available else "当前 Python 环境未检测到 GPU，可使用检索预览或本地 Ollama。"
    st.html('<div class="sidebar-label">回答方式</div>')
    mode = st.radio("回答方式", ["仅检索预览", "模型问答"], index=1 if generation_available else 0,
                    key="mode", label_visibility="collapsed", on_change=reset_results)
    st.caption("检索预览直接查阅指南；模型问答会根据指南整理操作建议。")
    st.html(f'<div class="runtime-status"><strong><span class="status-dot"></span>{status_title}</strong>'
            f'<small>{escape(status_detail)}</small></div>')
    if mode == "模型问答" and not generation_available:
        st.info("请检查上方的模型运行方式及服务状态，也可以切回检索预览。")
    st.html('<div class="sidebar-label">可以查阅的内容</div>')
    st.html("".join(
        f'<div class="sidebar-topic">{icon(symbol, 18)}<span>{label}</span></div>'
        for symbol, label in [("terminal", "Python 与 GPU 环境"), ("globe", "启动网页与端口访问"), ("save", "保存文件与管理会话")]
    ))
    with st.expander("演示参数", icon=":material/tune:"):
        method = st.selectbox("检索器", list(LABELS), format_func=LABELS.get, key="retriever-method", on_change=reset_results)
        k = st.select_slider("Top-k：最多检索片段数", options=[1, 3, 6, 10], value=3, on_change=reset_results)
        comparison = st.selectbox("对比方式", ["单次检索", "对比三种检索器（只检索）", "对比 Top-k 3 / 6"], key="comparison", on_change=reset_results)
        compare_retrievers = comparison == "对比三种检索器（只检索）"
        paired = comparison == "对比 Top-k 3 / 6"
        defaults = {"tfidf": 0.04, "bm25": 0.0, "dense": 0.0}
        min_score = st.number_input("最低分筛选门槛", min_value=0.0, value=defaults[method], step=0.01,
                                    key="threshold-" + method, disabled=compare_retrievers, on_change=reset_results)
        st.caption("三种检索器分别使用默认门槛。E5 默认保留 Top-k；语义分数尚未校准，需核对原文。首次使用会下载并编码文档。" if compare_retrievers or method == "dense" else "低于门槛的片段会被过滤。不同算法的分数不在同一个尺度。")
    st.html(f'<div class="sidebar-footnote">{len(data["documents"])} 页官方文档＋本项目说明<br>共 {len(data["passages"])} 个片段 / {escape(data["updated"])}</div>')
    if not data["complete_aup"]:
        st.warning("官方文档尚未完整准备，当前只使用已导入的资料。")
        if st.button("准备官方文档库", use_container_width=True):
            try:
                from prepare_knowledge import prepare
                with st.spinner("下载清单中的官方文档并切分…"):
                    prepare()
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    st.link_button("AUP 官方使用指南", OFFICIAL_GUIDE, icon=":material/open_in_new:", use_container_width=True)

badge = "检索预览" if mode == "仅检索预览" else (
    ("本地模型问答" if backend == "ollama" else "GPU 模型问答") if generation_available else "模型问答 · 待连接")
st.html(f'<div class="page-meta"><span class="page-meta-label">AMD AUP Learning Cloud 入门指南</span>'
        f'<span class="mode-badge">{icon("search" if mode == "仅检索预览" else "cloud", 15)}{badge}</span></div>')
st.html('<h1>云端开发，下一步怎么做？</h1><p class="page-intro">从配置 Python 到打开网页，描述你遇到的问题，找到操作指南和回答依据。</p>')

ask_tab, guides_tab = st.tabs(["问题解答", f"资料库 · {len(data['passages'])} 段"])
with ask_tab:
    with st.container(key="question-panel"):
        st.html(f'<div class="question-title">{icon("terminal", 22)}<h2>你遇到了什么问题？</h2></div>')
        with st.form("question-form", border=False):
            question = st.text_area("你的问题", key="question", height=130, label_visibility="collapsed",
                                    placeholder="粘贴报错，或描述你卡住的步骤。\n例如：Streamlit 已经启动，为什么浏览器打不开？")
            hint_column, action_column = st.columns([1.5, 1], vertical_alignment="center")
            with hint_column:
                hint = "同一问题、同一资料库，只比较检索结果" if compare_retrievers else ("直接查阅原文，附来源链接" if mode == "仅检索预览" else "首次问答需要加载模型，请稍候")
                st.html(f'<div class="question-hint">{hint}</div>')
            with action_column:
                single = st.form_submit_button(
                    "比较检索效果" if compare_retrievers else ("获取操作建议" if mode == "模型问答" else "查找相关资料"),
                    type="primary", icon=":material/search:", use_container_width=True,
                    disabled=mode == "模型问答" and not generation_available and not compare_retrievers,
                )

    if single:
        reset_results()
        if not question.strip():
            st.warning("请先输入问题，也可以点击下方常见问题填入。")
        elif mode == "模型问答" and not model_id.strip() and not compare_retrievers:
            st.warning("请在侧栏的“模型与运行环境”中填写模型名称。")
        else:
            try:
                generator = None
                methods = list(LABELS) if compare_retrievers else [method]
                active_retrievers = {}
                for current_method in methods:
                    with st.spinner(f"准备 {LABELS[current_method]}，首次可能下载模型和构建索引…"):
                        active_retrievers[current_method] = retrieval_resource(
                            data_revision, current_method, None if compare_retrievers else min_score)
                retriever = active_retrievers[method] if not compare_retrievers else None
                if mode == "模型问答" and not compare_retrievers and retriever.search(question, k):
                    with st.spinner("正在加载并预热模型，首次运行需要一些时间…"):
                        generator = model_resource(backend, model_id.strip(), base_url, num_ctx)
                with st.spinner("正在查阅指南…"):
                    results = []
                    selected = next((q for q in data["questions"] if q["question"] == question.strip()), None)
                    combinations = [(name, size) for name in methods for size in ([3, 6] if paired else [k])]
                    for current_method, top_k in combinations:
                        result = run_query(active_retrievers[current_method], question, top_k, generator,
                                           profile=data["prompt_profile"], max_new_tokens=data["max_new_tokens"],
                                           knowledge_base=data["id"])
                        result["recorded_at"] = datetime.now(timezone.utc).isoformat()
                        if selected:
                            result["question_id"] = selected["id"]
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
        st.html(f'<div class="empty-help">{icon("book", 20)}<span>先查资料，再看出处。<br>在“演示参数”切换检索器，在“资料库”阅读原文片段。</span></div>')
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
    st.html('<div class="library-heading"><h2>官方原文与项目说明</h2></div>')
    st.caption(f"已导入 {len(data['documents'])} 页官方文档；AUP 用户指南{'已完整导入' if data['complete_aup'] else '尚未完整导入'}。原文按章节切分，旧版 40 条 FAQ 不参与检索。")
    if data["failures"]:
        with st.expander("未成功导入的页面"):
            st.json(data["failures"])
    guide_search = st.text_input("搜索指南", placeholder="输入标题、报错或关键词，例如：端口占用、git pull、模型缓存", key="guide-search")
    categories = list(dict.fromkeys(p["category"] for p in data["passages"]))
    category = st.pills("筛选指南主题", ["全部"] + categories, default="全部", key="guide-category",
                        label_visibility="collapsed") or "全部"
    terms = guide_search.casefold().split()
    passages = [p for p in data["passages"]
                if (category == "全部" or p["category"] == category)
                and all(term in (p["title"] + " " + p["keywords"] + " " + p["text"]).casefold()
                        for term in terms)]
    page_count = max(1, (len(passages) + 19) // 20)
    page = st.selectbox("资料列表页码", range(1, page_count + 1), format_func=lambda n: f"第 {n} / {page_count} 页", key=f"library-page-{category}-{guide_search}")
    st.html(f'<div class="library-count">{escape(category)} / {len(passages)} 个片段，每页 20 个</div>')
    if not passages:
        st.info("没有符合条件的指南。试试其他关键词，或将主题切换为“全部”。")
    with st.container(key="guide-index"):
        for passage in passages[(page - 1) * 20:page * 20]:
            with st.expander(passage["title"]):
                st.caption(passage["category"])
                show_passage(passage)

st.html('<footer class="page-footer"><span>课程演示项目，非 AUP 官方客服。资源与账号状态以平台为准。</span>'
        f'<a href="{REPOSITORY}" target="_blank" rel="noopener noreferrer">GitHub 开源项目</a></footer>')
