"""
streamlit frontend for the sentiment engine.
type text, get sentiment analysis.
"""

import streamlit as st
from core import SentimentEngine, SentimentConfig

st.set_page_config(page_title="Sentiment Analysis", layout="centered")


@st.cache_resource
def load_engine():
    engine = SentimentEngine(SentimentConfig())
    engine.load()
    return engine


st.title("Sentiment Analysis")
st.caption("DistilBERT · 67M parameters · fine-tuned on SST-2")

engine = load_engine()

tab_single, tab_batch = st.tabs(["single analysis", "batch mode"])

with tab_single:
    text = st.text_area(
        "enter text to analyze", height=150,
        placeholder="I really enjoyed this movie, the acting was fantastic!",
    )

    if st.button("analyze", type="primary") and text.strip():
        result = engine.analyze(text)

        col1, col2, col3 = st.columns(3)
        label_color = "green" if result.label == "POSITIVE" else "red"
        col1.markdown(f"### :{label_color}[{result.label}]")
        col2.metric("confidence", f"{result.score:.1%}")
        col3.metric("inference", f"{result.inference_ms:.0f}ms")

        st.progress(result.score, text=f"{result.score:.2%} confidence")

with tab_batch:
    st.write("enter multiple texts, one per line:")
    batch_text = st.text_area(
        "texts", height=200,
        placeholder="this product is amazing\nterrible experience, would not recommend\npretty decent overall",
    )

    if st.button("analyze batch", type="primary") and batch_text.strip():
        texts = [t.strip() for t in batch_text.split("\n") if t.strip()]

        if len(texts) > engine.config.max_batch_size:
            st.error(f"max {engine.config.max_batch_size} texts at a time")
        else:
            results, total_ms = engine.analyze_batch(texts)

            pos_count = sum(1 for r in results if r.label == "POSITIVE")
            avg_conf = sum(r.score for r in results) / len(results)

            col1, col2, col3 = st.columns(3)
            col1.metric("positive", f"{pos_count / len(results):.0%}")
            col2.metric("avg confidence", f"{avg_conf:.2f}")
            col3.metric("total texts", len(texts))

            st.divider()
            for r in results:
                icon = "+" if r.label == "POSITIVE" else "-"
                st.write(f"`[{icon}] {r.score:.2f}` — {r.text}")
