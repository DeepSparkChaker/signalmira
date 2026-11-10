"""SignalMira local Streamlit dashboard.

Database browsing only reads saved reviews and enrichments; it never calls AI.
Pasted-review analysis runs NER, optional catalog matching and topic preview,
without saving the pasted text to PostgreSQL.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from frontend import service
from frontend.prepare_models import ner_ready
from frontend.retrieval_lab import LabConfig, EMBEDDING_REVISION, SAMPLE_QUERY, SAMPLE_REVIEW_ID, SAMPLE_REVIEW_TEXT, warm_selected, search_lab


st.set_page_config(page_title="SignalMira", page_icon="◈", layout="wide")

NER_DESCRIPTIONS = {
    "balanced": "GLiNER2.5 Small — lighter model for custom English entities",
    "fast": "spaCy — fastest CPU option, mainly standard English entities",
    "accurate": "GLiNER2.5 Base — larger English model, potentially more accurate",
    "multilingual": "GLiNER2.5 Multi — for English, French, Spanish, Arabic, Chinese, etc.",
}


def review_metadata(row: dict) -> dict:
    """Original optional review fields are stored in feedback_reviews.attributes."""
    attrs = row.get("attributes") or {}
    if not isinstance(attrs, dict):
        return {}
    nested = attrs.get("attributes")
    # Support both direct metadata and a nested "attributes" dict.
    return {**nested, **attrs} if isinstance(nested, dict) else attrs


def metadata_text(row: dict, *keys: str) -> str | None:
    """Return a saved metadata value, never invent a missing title or label."""
    attrs = review_metadata(row)
    for key in keys:
        value = attrs.get(key)
        if isinstance(value, (str, int, float, bool)) and str(value).strip():
            return str(value)
    return None


def review_title(row: dict) -> str | None:
    return metadata_text(row, "title", "review_title", "reviewTitle", "headline")


def review_label(row: dict) -> str | None:
    # Source/dataset label, NOT the NER entity type.
    return metadata_text(row, "label", "sentiment_label", "sentiment", "class")


def ner_entities(row: dict) -> list[dict]:
    entities = row.get("entities") or []
    return [e for e in entities if isinstance(e, dict)] if isinstance(entities, list) else []


def ner_names_and_types(row: dict) -> str:
    mentions = ner_entities(row)
    if not mentions:
        return "No entities" if row.get("ner_done") else "Not processed"
    return "; ".join(dict.fromkeys(
        f'{e.get("text", "—")} ({e.get("label", "—")})'
        for e in mentions
    ))


@st.cache_resource(max_entries=2, show_spinner="Loading NER model once…")
def ner_worker(profile: str, device: str):
    from enrichment.ner import Config, NER
    return NER(Config(profile=profile, device=device, offline=True)), Lock()


@st.cache_resource(max_entries=1, show_spinner="Starting opinion model…")
def opinion_worker():
    from agent import build_extractor
    return build_extractor(), Lock()


@st.cache_data(ttl=30, show_spinner=False)
def get_overview(tenant: str):
    return service.overview(tenant)


@st.cache_resource(max_entries=1, show_spinner=False)
def retrieval_worker(config: LabConfig):
    """One global, thread-safe embedder + selected reranker per app process."""
    return warm_selected(config)


st.title("◈ SignalMira")
st.caption("Customer feedback → clear, actionable signals")

options = service.tenants()
if not options:
    st.error("Configure SIGNALMIRA_DEMO_TENANTS")
    st.stop()

tenant = st.selectbox("Workspace", options)
st.caption("Local preview · tenant dropdown is not authentication")

# Keep each workspace's cached UI results separate.
if st.session_state.get("_workspace") != tenant:
    for key in ("stats", "rows", "analysis", "review_input", "retrieval_output"):
        st.session_state.pop(key, None)
    st.session_state["_workspace"] = tenant

# First Streamlit page execution blocks here until the selected GPU models
# are warm; subsequent reruns/sessions reuse them. Strict HTTP readiness
# before the first connection requires FastAPI lifespan (future milestone).
lab_models = None
lab_problem = None
lab_enabled = os.getenv("SIGNALMIRA_RETRIEVAL_LAB", "1") == "1"
if lab_enabled:
    try:
        model_choice = os.getenv("SIGNALMIRA_RERANKER_MODEL", "qwen").strip().lower()
        lab_config = LabConfig(
            embedding_revision=os.getenv("SIGNALMIRA_EMBEDDING_SHA", "") or EMBEDDING_REVISION,
            reranker_model=model_choice,
            reranker_revision=os.getenv("SIGNALMIRA_RERANKER_SHA", ""),
            device=os.getenv("SIGNALMIRA_MODEL_DEVICE", "cuda"),
            timeout_s=float(os.getenv("SIGNALMIRA_RERANK_TIMEOUT", "10")),
        )
        with st.spinner(f"Warming Qwen embeddings + {model_choice} reranker (selected only)…"):
            lab_models = retrieval_worker(lab_config)
    except Exception as exc:
        lab_problem = str(exc)
        # Existing review browser and NER panels remain available.
        st.warning(f"Retrieval Lab disabled: {lab_problem}")

if os.getenv("SIGNALMIRA_PREWARM", "0") == "1":
    try:
        ner_worker("balanced", "auto")[0].warmup(count=1)
    except Exception as exc:
        st.warning(f"Default NER prewarm failed: {exc}")

# No PostgreSQL call until the user requests feedback (except optional prewarm).
if st.button("Show my feedback", type="primary"):
    try:
        st.session_state["stats"] = get_overview(tenant)
        st.session_state["rows"] = service.latest_reviews(tenant)
    except Exception as exc:
        st.error(f"Database unavailable: {exc}")

stats = st.session_state.get("stats")
if stats:
    a, b, c, d = st.columns(4)
    a.metric("Reviews", stats["total"])
    b.metric("Sources", stats["sources"])
    c.metric("Updated in 7 days", stats["recent"])
    d.metric("NER processed", stats["ner_processed"])

    st.subheader("Find customer reviews")
    st.caption(
        "Search ALL stored reviews for this workspace, then display up to "
        "10 newest matches. Missing NER/topic data does not hide a review "
        "unless you explicitly filter by that enrichment."
    )

    with st.form("filters"):
        c1, c2, c3 = st.columns(3)
        source = c1.selectbox(
            "Review source", ["", *stats["source_options"]],
            format_func=lambda x: x or "All sources",
        )
        entity_type = c2.selectbox(
            "NER entity type", ["", *stats["entity_labels"]],
            format_func=lambda x: x or "All entity types",
            help="Example: product, brand, organization, product feature",
        )
        choices = {"All topics": None}
        for item in stats["topics"]:
            choices[f'{item["name"]} (#{item["topic_id"]})'] = item["topic_id"]
        topic_name = c3.selectbox("Saved topic", list(choices))

        c4, c5 = st.columns(2)
        keyword = c4.text_input(
            "Search original review text",
            max_chars=120,
            placeholder="battery, delivery, iPhone…",
            help="Finds the phrase in the original text; does NOT run AI.",
        )
        entity_name = c5.text_input(
            "Search extracted NER name",
            max_chars=100,
            placeholder="iPhone, Amazon, Google, Pixel 9…",
            help=(
                "Searches the saved 'text' in feedback_entities, not the "
                "full review. If an entity type is also chosen, BOTH "
                "filters must match the same extracted entity."
            ),
        )
        only_ner = st.checkbox("Only reviews with completed, current NER", value=False)
        search = st.form_submit_button("Apply filters", type="primary")

    if search:
        try:
            st.session_state["rows"] = service.latest_reviews(
                tenant,
                source=source,
                keyword=keyword.strip(),
                entity=entity_type,
                entity_name=entity_name.strip(),
                topic_id=choices[topic_name],
                has_ner=only_ner,
            )
        except Exception as exc:
            st.error(f"Could not search: {exc}")

    rows = st.session_state.get("rows", [])
    if stats["taxonomy"]:
        status = "approved" if stats["taxonomy"]["approved"] else "candidate (not approved)"
        st.caption(f'Topic taxonomy: {status} · {stats["taxonomy"]["model_name"]}')
    else:
        st.caption("No active or pending taxonomy. Saved topic filtering is unavailable.")

    if rows:
        st.caption(f"Showing {len(rows)} latest matching review(s) · maximum 10 per search")
        st.caption(
            "Review title and dataset label come from saved attributes when available. "
            "NER names and types come from the saved extraction results."
        )

        # Data already comes from service.latest_reviews(): no additional queries or models.
        display_rows = []
        for r in rows:
            entities = ner_entities(r)
            display_rows.append({
                "Source": r["source"],
                "Review ID": r["review_id"],
                "Review title": review_title(r) or "—",
                "Review label": review_label(r) or "—",
                "Full review text": r["text"],
                "NER names & labels": ner_names_and_types(r),
                "NER types": ", ".join(dict.fromkeys(
                    str(e.get("label", "—")) for e in entities
                )) or ("No entities" if r["ner_done"] else "Not processed"),
                "Topic": r["topic_name"] or "Not assigned",
                "NER model": ", ".join(dict.fromkeys(
                    str(e.get("model", "—")) for e in entities if e.get("model")
                )) or ("Processed; no entities" if r["ner_done"] else "—"),
                "Topic model": r["topic_model"] or "—",
            })

        st.dataframe(
            display_rows,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Full review text": st.column_config.TextColumn(
                    "Full review text", width="large"
                ),
                "NER names & labels": st.column_config.TextColumn(
                    "NER names & labels", width="large",
                    help="Saved named-entity mention followed by its NER category",
                ),
                "Review title": st.column_config.TextColumn("Review title", width="medium"),
            },
        )

        st.subheader("Review details")
        for r in rows:
            title = review_title(r)
            original_label = review_label(r)
            entities = ner_entities(r)
            expander_title = (title or r["text"][:65] or r["review_id"]).replace("\n", " ")

            with st.expander(f'{expander_title[:80]} · {r["source"]}/{r["review_id"]}'):
                st.write("**Review title:**", title or "Not provided in source data")
                st.write("**Review label (original dataset):**", original_label or "Not provided")
                st.write("**Full review text**")
                st.write(r["text"])
                st.caption(
                    f'Company: {r["company"] or "—"} · '
                    f'Product: {r["product_id"] or "—"} · '
                    f'Date: {r["review_date"] or "—"}'
                )

                st.write("**NER highlights in original review**")
                st.html(service.highlights(r["text"], entities))

                if r["ner_done"]:
                    st.write("**Extracted entity names and labels**")
                    if entities:
                        st.dataframe(
                            [{
                                "Entity name": e.get("text"),
                                "Entity label": e.get("label"),
                                "NER score": e.get("confidence"),
                                "NER model": e.get("model"),
                                "Catalog match": e.get("catalog"),
                                "Link score": e.get("catalog_score"),
                            } for e in entities],
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.caption("NER processed this review but detected no entities.")
                    st.caption(f'NER model revision: {r["ner_revision"] or "unknown"}')
                else:
                    st.caption("NER not processed or saved result is outdated")

                st.write("**Saved topic:**", r["topic_name"] or "Not assigned")
                if r["topic_model"] and r["topic_score"] is not None:
                    st.caption(
                        f'Topic model: {r["topic_model"]} · '
                        f'cosine similarity: {r["topic_score"]:.3f}'
                    )
                st.caption("Original source metadata:")
                st.json(r["attributes"] or {})
    else:
        st.info("No stored reviews match these filters.")

st.divider()
st.subheader("Analyze one review")
st.caption("Preview only. The pasted review and its predictions are NOT saved to PostgreSQL.")
with st.form("review_lab"):
    text = st.text_area(
        "Customer feedback", max_chars=service.MAX_CHARS, height=100,
        key="review_input",
        placeholder="My Pixel 9 camera is excellent but the battery drains overnight…",
    )
    with st.expander("Model options"):
        profile = st.selectbox(
            "NER accuracy / speed",
            ["balanced", "fast", "accurate", "multilingual"],
            format_func=lambda p: NER_DESCRIPTIONS[p],
        )
        device = st.selectbox("Hardware", ["auto", "cpu", "cuda"])
        opinions = st.checkbox("Also extract opinions (Qwen)", value=False)
    submit = st.form_submit_button("Analyze feedback", type="primary")

if submit:
    st.session_state.pop("analysis", None)
    if not text.strip():
        st.warning("Enter a review first.")
    else:
        ready, reason = ner_ready(profile)
        if not ready:
            st.error(f"NER model is not installed locally: {reason}")
            st.code(
                f"uv run --extra ner python -m frontend.prepare_models "
                f"--download --ner {profile} --tenant {tenant}",
                language="bash",
            )
        else:
            try:
                model, lock = ner_worker(profile, device)
                qwen = None
                if opinions:
                    try:
                        qwen = opinion_worker()
                    except Exception as exc:
                        st.warning(f"Qwen unavailable: {exc}")
                with st.spinner("Extracting signals…"):
                    result = service.analyze(
                        tenant, text, model, lock, device=device, opinion_model=qwen
                    )
                st.session_state["analysis"] = (
                    tenant, text, profile, device, opinions, result
                )
            except Exception as exc:
                st.error(f"Unable to analyze: {exc}")

state = st.session_state.get("analysis")
if state and state[:5] == (tenant, text, profile, device, opinions):
    out = state[5]
    for message in out["warnings"]:
        st.warning(message)
    st.write("**Named entities** ·", out["ner_model"])
    st.html(service.highlights(text, out["entities"]))
    if out["entities"]:
        st.dataframe(
            [{
                "Mention": e["text"],
                "Type": e["label"],
                "Score": e.get("confidence"),
                "Catalog": (e.get("catalog_link") or {}).get("canonical_name"),
            } for e in out["entities"]], hide_index=True,
        )
    st.write("**Topic:**", out["topic"]["name"] if out["topic"] else "Not available")
    if out["topic"]:
        st.caption(
            f'{"Approved" if out["topic"]["approved"] else "Candidate"} · '
            f'{out["topic"]["model"]} · cosine {out["topic"]["score"]:.3f}'
        )
    if out["opinions"]:
        st.write("**Opinion analysis**", out["opinions"].get("overall_sentiment"))
        st.dataframe(out["opinions"].get("opinions", []), hide_index=True)
    st.caption("NER scores and cosine similarity are not calibrated probabilities.")

st.caption("Local MVP · Existing PostgreSQL/NLP engines · Future service layer: FastAPI")




st.divider()
st.subheader("Retrieval Lab · BM25 + dense + RRF + reranker")
st.caption("Read-only local evaluation. The question is not saved. Models are shared across reruns; database reads are tenant-scoped.")
if not lab_enabled:
    st.info("Set SIGNALMIRA_RETRIEVAL_LAB=1 to enable this lab.")
elif lab_models is None:
    st.error("Retrieval models unavailable. Set SIGNALMIRA_RERANKER_SHA to the selected model's pinned SHA and check local model snapshots.")
else:
    embedder, ranker, warmup = lab_models
    st.caption(
        f"Warm: Qwen embedding + {lab_config.reranker_model} | "
        f"Startup: embed {warmup['embedding_warmup_ms'] / 1000:.1f}s, "
        f"rerank {warmup['reranker_warmup_ms'] / 1000:.1f}s"
    )
    with st.form("retrieval_lab_form"):
        rag_query = st.text_input("Search customer feedback", value=SAMPLE_QUERY, max_chars=2048)
        rc1, rc2, rc3 = st.columns(3)
        lexical = rc1.selectbox("Lexical search", ["bm25", "fts"])
        retrieval_mode = rc2.selectbox("Dense mode", ["exact", "hnsw"])
        only_source = rc3.text_input("Source filter (optional)", value="uci-amazon", max_chars=120)
        accept_idf = st.checkbox("Allow shared BM25 corpus statistics (LOCAL demo only)", value=False,
                                 help="Required for the pg_textsearch backend; not strict corpus-level tenant isolation.")
        run_retrieval = st.form_submit_button("Search and compare top 8", type="primary")
    if run_retrieval:
        st.session_state.pop("retrieval_output", None)
        try:
            service.guard(tenant)
            if lexical == "bm25" and not accept_idf:
                raise ValueError("Enable the shared BM25 statistics checkbox, or select FTS.")
            with st.spinner("Retrieving and reranking…"):
                with service.connect() as conn:
                    report = search_lab(
                        conn, tenant, rag_query, embedder, ranker,
                        lexical_backend=lexical, allow_shared_idf=accept_idf,
                        mode=retrieval_mode, source=only_source.strip() or None,
                    )
            st.session_state["retrieval_output"] = (tenant, report)
        except Exception as exc:
            st.error(f"Retrieval failed: {type(exc).__name__}: {exc}")
    saved = st.session_state.get("retrieval_output")
    if saved and saved[0] == tenant:
        report = saved[1]
        if report["status"] == "rrf_fallback":
            st.warning(f"Reranker fallback to RRF: {report['reason']}")
        a, b, c, d = st.columns(4)
        a.metric("Embedding", f"{report['embedding_ms']:.0f} ms")
        b.metric("PostgreSQL hybrid", f"{report['postgres_ms']:.0f} ms")
        c.metric("Rerank", f"{report['rerank_ms']:.0f} ms")
        d.metric("Candidates", report["candidate_count"])
        st.caption(f"Status: {report['status']} · measured total {report['total_ms']:.0f} ms (excludes Streamlit render)")
        reference_found = any(row.get("review_id") == SAMPLE_REVIEW_ID for row in report["results"])
        if report["query"] == SAMPLE_QUERY:
            if reference_found:
                st.success(f"Reference review {SAMPLE_REVIEW_ID} appears in the top 8.")
            else:
                st.info(f"Reference review {SAMPLE_REVIEW_ID} is not in this top 8; inspect filters/indexes and rankings.")
            st.caption(f"Reference review: ‘{SAMPLE_REVIEW_TEXT}’")
        left, right = st.columns(2)
        def rows_for_display(items):
            return [{
                "RRF rank": hit.get("rrf_rank"),
                "Review": hit.get("review_id"),
                "Text": hit.get("chunk_text"),
                "Rerank score": round(float(hit["rerank_score"]), 3)
                                if hit.get("rerank_score") is not None else None,
                "Dense rank": hit.get("dense_rank"),
                "BM25 rank": hit.get("sparse_rank"),
            } for hit in items]
        with left:
            st.markdown("**Original RRF top 8**")
            st.dataframe(rows_for_display(report["rrf_top8"]), hide_index=True, use_container_width=True)
        with right:
            st.markdown("**After Qwen/BGE reranking — top 8**")
            st.dataframe(rows_for_display(report["results"]), hide_index=True, use_container_width=True)
        with st.expander("Inspect exact source chunks and IDs"):
            for i, hit in enumerate(report["results"], start=1):
                st.markdown(f"**{i}. {hit.get('review_id', '—')}** · chunk `{hit.get('chunk_id', '')}`")
                st.write(hit.get("chunk_text", ""))
                st.caption(f"RRF rank: {hit.get('rrf_rank')} · RRF score: {hit.get('rrf_score')} · "
                           f"rerank score: {hit.get('rerank_score', '—')}")
