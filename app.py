"""Small AUP onboarding FAQ demo; model inference runs on the cloud GPU."""

import importlib.util
import os
from datetime import datetime, timezone

import streamlit as st

from rag import DEFAULT_MODEL, Generator, Retriever, environment, export_csv, load_data, run_query

st.set_page_config(page_title="AUP 入门助手", page_icon="🧭", layout="wide")
st.markdown("""
<style>
.block-container {max-width:1120px;padding-top:4rem;padding-bottom:3rem}
div[data-testid="stMetric"] {background:white;border:1px solid #dfe8e2;border-radius:12px;padding:14px}
.eyebrow {color:#236b60;font-size:12px;letter-spacing:2px;font-weight:700}
.intro {color:#5b7068;font-size:17px;line-height:1.7;margin-bottom:20px}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def resources():
    data = load_data()
    return data, Retriever(data["passages"], **data.get("retriever", {}))


@st.cache_resource(max_entries=1)
def model_resource(model_id):
    return Generator(model_id)


def show_passage(passage):
    st.caption(passage["scope"])
    st.markdown(passage["text"])
    for source in passage["sources"]:
        st.markdown(f"[{source['title']}]({source['url']})")
        st.caption(f"{source['kind']} · 核对于 {source['checked_on']}")


data, retriever = resources()
with st.sidebar:
    st.markdown("### 🧭 AUP 入门助手")
    st.caption("AMD 云端课程 demo")
    st.divider()
    has_torch = importlib.util.find_spec("torch") is not None
    mode = st.radio("运行方式", ["仅检索预览", "模型问答"], index=1 if has_torch else 0)
    st.caption("预览可直接查看指南；模型问答由云端 GPU 生成建议。")
    st.divider()
    st.markdown("**这份指南能帮你做什么**")
    st.write("准备 Python 环境\n\n启动和访问网页\n\n保存结果与管理会话")
    st.caption(f"{len(data['passages'])} 条指南 · {len(data['questions'])} 个示例 · {data['updated']}")
    st.markdown("[AUP 官方使用指南](https://amdresearch.github.io/aup-learning-cloud/user-guide/platform-basics.html)")
    with st.expander("模型与运行环境"):
        model_id = st.text_input("生成模型", value=os.getenv("MODEL_ID", DEFAULT_MODEL),
                                 disabled=mode == "仅检索预览")
        st.json(environment())

st.markdown('<div class="eyebrow">AUP / QUICK START COMPANION</div>', unsafe_allow_html=True)
st.title("AUP 云端开发入门助手")
st.markdown('<div class="intro">从环境配置到打开网页，找到下一步怎么做，也能查看建议的出处。</div>', unsafe_allow_html=True)
st.caption("基于官方文档与本项目部署记录整理的课程演示。适合入门查阅，具体资源和账号状态请以平台页面为准。")

ask_tab, guides_tab = st.tabs(["💬 提问", "📚 浏览 8 条指南"])
with ask_tab:
    choices = ["自定义问题"] + [q["question"] for q in data["questions"]]
    index = st.selectbox("从一个常见问题开始", range(len(choices)), format_func=lambda i: choices[i], index=1)
    selected = data["questions"][index - 1] if index else None
    question = st.text_area("你的问题", value=selected["question"] if selected else "",
                            key=f"question_{index}", height=95,
                            placeholder="例如：云端已经启动了 Streamlit，为什么 localhost 打不开？")
    with st.expander("可选演示：检索数量与开销"):
        k = st.select_slider("最多检索几条指南", options=[3, 6], value=3)
        paired = st.checkbox("同时比较最多 3 条与 6 条指南", value=False)
        st.caption("只选相关性达到门槛的指南，实际数量可能更少。对比仅用于观察这个 demo 的运行开销。")
    single = st.button("获取操作建议" if mode == "模型问答" else "查找相关指南",
                       type="primary", use_container_width=True)

    if single:
        st.session_state.pop("results", None)
        if not question.strip():
            st.warning("请先输入一个问题。")
        else:
            try:
                generator = None
                # Do not download/load a model when no material passes the basic filter.
                if mode == "模型问答" and retriever.search(question, k):
                    with st.spinner("加载并预热模型，首次运行需要一些时间…"):
                        generator = model_resource(model_id.strip())
                with st.spinner("正在查阅指南…"):
                    results = []
                    for top_k in ([3, 6] if paired else [k]):
                        result = run_query(retriever, question, top_k, generator,
                                           profile=data["prompt_profile"],
                                           max_new_tokens=data["max_new_tokens"],
                                           knowledge_base=data["id"])
                        result["recorded_at"] = datetime.now(timezone.utc).isoformat()
                        if selected and question.strip() == selected["question"]:
                            result["question_id"] = selected["id"]
                            result["reference_answers"] = " | ".join(selected["answers"])
                            result["reference_passage_retrieved"] = any(
                                p["id"] == selected["passage_id"] for p in result["passages"]
                            )
                        results.append(result)
                    st.session_state["results"] = results
            except Exception as exc:
                st.error(f"本次运行未完成：{exc}")

    results = st.session_state.get("results", [])
    if results:
        st.divider()
        st.subheader("查阅结果")
        st.caption(f"本次问题：{results[0]['question']}")
        if results[0]["mode"] == "retrieval_preview" and results[0]["status"] != "no_evidence":
            st.info("当前是检索预览，可以展开下方指南。切换到“模型问答”后会生成操作建议。")
        columns = st.columns(len(results))
        for column, result in zip(columns, results):
            with column:
                if len(results) > 1:
                    st.markdown(f"### 最多检索 {result['top_k']} 条")
                if result["notice"]:
                    st.warning(result["notice"])
                if result["answer"] is not None:
                    st.markdown(result["answer"])
                if result["passages"]:
                    st.markdown(f"**相关指南 · {result['retrieved_count']} 条**")
                for i, passage in enumerate(result["passages"], 1):
                    with st.expander(f"[P{i}] {passage['title']}"):
                        show_passage(passage)
                with st.expander("运行记录与核对"):
                    a, b, c = st.columns(3)
                    a.metric("总耗时", f"{result['total_seconds']:.3f} s")
                    b.metric("输入 token", result["input_tokens"] if result["input_tokens"] is not None else "—")
                    c.metric("输出 token", result["output_tokens"] if result["output_tokens"] is not None else "—")
                    if result.get("reference_answers"):
                        st.write("核对要点：" + result["reference_answers"])
                        st.caption("参考指南已检索到" if result["reference_passage_retrieved"] else "未检索到参考指南")
                    st.caption("计时不含首次下载、加载、预热和网页渲染。仅检索时不生成答案、不计模型 token。")
        st.download_button("下载本次结果 CSV", export_csv(results), file_name="aup_demo_results.csv", mime="text/csv")
    else:
        st.info("先选择一道示例题，或描述你遇到的入门问题。每条指南都附有来源和适用范围。")

with guides_tab:
    st.write("这些是为本课程 demo 编写的简短操作指南。可以直接阅读，也可以通过提问查找。")
    for category in dict.fromkeys(p["category"] for p in data["passages"]):
        st.markdown(f"### {category}")
        for passage in data["passages"]:
            if passage["category"] == category:
                with st.expander(passage["title"]):
                    show_passage(passage)

st.caption("8 条固定指南 · 单轮资料问答 · 不连接账号或执行命令 · 课程 demo，非 AUP 官方客服")
