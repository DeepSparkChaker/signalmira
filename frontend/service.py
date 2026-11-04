"""Read-only, tenant-scoped SignalMira service. Reuse from FastAPI later.

Never infer tenant identity from customer text: the caller supplies an authorized tenant.
For now `SIGNALMIRA_DEMO_TENANTS` is a LOCAL allowlist, not authentication.
"""
from __future__ import annotations

import os
from html import escape
from threading import Lock
from typing import Any

MAX_CHARS = 8000
TOPIC_LOCK = Lock()
COLORS = {"company": "#dbeafe", "brand": "#ede9fe", "product": "#d1fae5",
          "product feature": "#ffedd5", "organization": "#e0f2fe", "person": "#fce7f3"}


def tenants() -> tuple[str, ...]:
    return tuple(dict.fromkeys(t.strip() for t in
                 os.getenv("SIGNALMIRA_DEMO_TENANTS", "demo").split(",") if t.strip()))


def guard(tenant: str) -> None:
    if tenant not in tenants():
        raise PermissionError("Tenant not allowed in this local demo")


def connect():
    from storage.postgres import connect as pg_connect
    return pg_connect(os.environ["DATABASE_URL"])


def exists(conn, table: str) -> bool:
    """Names are hardcoded by this module; never use values from customers."""
    return bool(conn.execute("SELECT to_regclass(%s) IS NOT NULL AS ok",
                             (f"public.{table}",)).fetchone()["ok"])


def _tables(conn) -> dict[str, bool]:
    """One round trip to determine which optional enrichment tables exist."""
    names = ("feedback_ner_state", "feedback_entities", "entity_links",
             "sm_topic_active", "sm_topic_discovery_state",
             "sm_topic_taxonomies", "sm_topic_assignments", "sm_topic_definitions")
    sql = "SELECT " + ",".join(f"to_regclass('public.{name}') IS NOT NULL AS {name}"
                              for name in names)  # hardcoded identifiers only
    return dict(conn.execute(sql).fetchone())


def _taxonomy(conn, tenant: str, tables: dict[str, bool]) -> dict | None:
    """Approved first. If none, preview pending taxonomy clearly as a candidate."""
    if tables["sm_topic_active"] and tables["sm_topic_taxonomies"]:
        row = conn.execute("""SELECT a.taxonomy_id,t.model_name
            FROM sm_topic_active a JOIN sm_topic_taxonomies t
            USING (tenant_id,taxonomy_id) WHERE a.tenant_id=%s""", (tenant,)).fetchone()
        if row:
            return {**row, "approved": True}
    if tables["sm_topic_discovery_state"] and tables["sm_topic_taxonomies"]:
        row = conn.execute("""SELECT t.taxonomy_id,t.model_name
            FROM sm_topic_discovery_state s JOIN sm_topic_taxonomies t
              ON t.tenant_id=s.tenant_id AND t.taxonomy_id=s.pending_taxonomy_id
            WHERE s.tenant_id=%s""", (tenant,)).fetchone()
        if row:
            return {**row, "approved": False}
    return None


def overview(tenant: str) -> dict:
    """Fast tenant counts and filter options. Does NOT load or run ML models."""
    guard(tenant)
    with connect() as conn:
        present = _tables(conn)
        stats = dict(conn.execute("""SELECT count(*) AS total,
            count(DISTINCT source) AS sources,
            count(*) FILTER (WHERE updated_at >= now() - interval '7 days') AS recent
            FROM feedback_reviews WHERE tenant_id=%s""", (tenant,)).fetchone())
        stats["source_options"] = [r["source"] for r in conn.execute(
            "SELECT DISTINCT source FROM feedback_reviews WHERE tenant_id=%s ORDER BY source", (tenant,))]
        stats["entity_labels"] = ([r["label"] for r in conn.execute(
            "SELECT DISTINCT label FROM feedback_entities WHERE tenant_id=%s ORDER BY label", (tenant,))]
            if present["feedback_entities"] else [])
        taxonomy = _taxonomy(conn, tenant, present)
        stats["taxonomy"] = taxonomy
        stats["topics"] = (list(conn.execute("""SELECT topic_id,name
            FROM sm_topic_definitions WHERE tenant_id=%s AND taxonomy_id=%s
            AND topic_id >= 0 ORDER BY name""", (tenant, taxonomy["taxonomy_id"])))
            if taxonomy and present["sm_topic_definitions"] else [])
        stats["ner_processed"] = (conn.execute("""SELECT count(*) AS n
            FROM feedback_ner_state s JOIN feedback_reviews f
            USING (tenant_id,source,review_id)
            WHERE f.tenant_id=%s AND s.content_hash=f.content_hash""",
            (tenant,)).fetchone()["n"] if present["feedback_ner_state"] else 0)
    return stats


def build_review_query(*, source: str = "", keyword: str = "", entity: str = "",
                       entity_name: str = "", topic_id: int | None = None, has_ner: bool = False,
                       tables: dict[str, bool], taxonomy_id=None) -> tuple[str, list]:
    """Filter ALL original tenant reviews, then enrich only the newest 10 matches.

    EXISTS filters intentionally require saved/current NER or topic rows only when
    requested. Optional NER, NED, and topic enrichments are LEFT JOINed afterwards.
    """
    where = ["f.tenant_id=%s"]
    params: list[Any] = []  # tenant is prepended by caller
    if source:
        where.append("f.source=%s")
        params.append(source)
    if keyword:
        where.append("position(lower(%s) in lower(f.text)) > 0")
        params.append(keyword)
    if has_ner and tables["feedback_ner_state"]:
        where.append("""EXISTS (SELECT 1 FROM feedback_ner_state s
            WHERE s.tenant_id=f.tenant_id AND s.source=f.source AND s.review_id=f.review_id
            AND s.content_hash=f.content_hash)""")
    if (entity or entity_name) and tables["feedback_entities"] and tables["feedback_ner_state"]:
        criteria = ["e.tenant_id=f.tenant_id", "e.source=f.source", "e.review_id=f.review_id",
                    "s.content_hash=f.content_hash"]
        if entity:
            criteria.append("e.label=%s")
            params.append(entity)
        if entity_name:
            criteria.append("position(lower(%s) in lower(e.text)) > 0")
            params.append(entity_name)
        where.append("EXISTS (SELECT 1 FROM feedback_entities e "
                     "JOIN feedback_ner_state s USING (tenant_id,source,review_id) WHERE "
                     + " AND ".join(criteria) + ")")
    if topic_id is not None and tables["sm_topic_assignments"] and taxonomy_id is not None:
        where.append("""EXISTS (SELECT 1 FROM sm_topic_assignments ta
            WHERE ta.tenant_id=f.tenant_id AND ta.source=f.source AND ta.review_id=f.review_id
              AND ta.content_hash=f.content_hash AND ta.taxonomy_id=%s AND ta.topic_id=%s)""")
        params.extend((taxonomy_id, topic_id))
    clauses = " AND ".join(where)
    sql = f"""WITH selected AS (
      SELECT f.tenant_id,f.source,f.review_id,f.text,f.company,f.product_id,
             f.review_date,f.updated_at,f.attributes,f.content_hash
      FROM feedback_reviews f WHERE {clauses}
      ORDER BY f.updated_at DESC,f.source,f.review_id LIMIT 10
    ) SELECT f.*"""
    if tables["feedback_ner_state"]:
        sql += ", s.model_revision AS ner_revision, (s.review_id IS NOT NULL) AS ner_done"
    else:
        sql += ", NULL::text AS ner_revision, false AS ner_done"
    if tables["feedback_entities"] and tables["feedback_ner_state"]:
        link_join = ("LEFT JOIN entity_links l ON l.tenant_id=e.tenant_id AND l.source=e.source "
                     "AND l.review_id=e.review_id AND l.mention_index=e.mention_index"
                     if tables["entity_links"] else "")
        link_value = ("'catalog',l.canonical_name,'catalog_score',l.score" if tables["entity_links"]
                      else "'catalog',NULL,'catalog_score',NULL")
        sql += f""", COALESCE(m.entities,'[]'::jsonb) AS entities
          FROM selected f
          LEFT JOIN feedback_ner_state s ON s.tenant_id=f.tenant_id
            AND s.source=f.source AND s.review_id=f.review_id AND s.content_hash=f.content_hash
          LEFT JOIN LATERAL (SELECT jsonb_agg(jsonb_build_object(
            'text',e.text,'label',e.label,'confidence',e.confidence,
            'model',e.model_name,'start',e.start_char,'end',e.end_char,
            {link_value}) ORDER BY e.mention_index) AS entities
            FROM feedback_entities e {link_join}
            WHERE e.tenant_id=f.tenant_id AND e.source=f.source AND e.review_id=f.review_id
              AND s.review_id IS NOT NULL) m ON true"""
    else:
        if tables["feedback_ner_state"]:
            sql += """, '[]'::jsonb AS entities FROM selected f
            LEFT JOIN feedback_ner_state s ON s.tenant_id=f.tenant_id
            AND s.source=f.source AND s.review_id=f.review_id AND s.content_hash=f.content_hash"""
        else:
            sql += ", '[]'::jsonb AS entities FROM selected f"
    if tables["sm_topic_assignments"] and tables["sm_topic_definitions"] and taxonomy_id:
        sql += """ LEFT JOIN LATERAL (SELECT a.topic_id,a.similarity,d.name,
               tx.model_name FROM sm_topic_assignments a
            JOIN sm_topic_definitions d ON d.tenant_id=a.tenant_id AND d.taxonomy_id=a.taxonomy_id
              AND d.topic_id=a.topic_id
            JOIN sm_topic_taxonomies tx ON tx.tenant_id=a.tenant_id AND tx.taxonomy_id=a.taxonomy_id
            WHERE a.tenant_id=f.tenant_id AND a.source=f.source AND a.review_id=f.review_id
              AND a.content_hash=f.content_hash AND a.taxonomy_id=%s LIMIT 1) t ON true"""
        sql = sql.replace("SELECT f.*", "SELECT f.*,t.topic_id,t.similarity AS topic_score,t.name AS topic_name,t.model_name AS topic_model", 1)
        params.append(taxonomy_id)
    else:
        sql = sql.replace("SELECT f.*", "SELECT f.*,NULL::int AS topic_id,NULL::float8 AS topic_score,NULL::text AS topic_name,NULL::text AS topic_model", 1)
    sql += " ORDER BY f.updated_at DESC,f.source,f.review_id"
    return sql, params


def latest_reviews(tenant: str, *, source: str = "", keyword: str = "",
                   entity: str = "", entity_name: str = "",
                   topic_id: int | None = None,
                   has_ner: bool = False) -> list[dict]:
    """Search all original reviews; return up to 10 newest matches with optional enrichments."""
    guard(tenant)
    if len(keyword) > 120 or len(entity_name) > 100:
        raise ValueError("Search phrase too long")
    with connect() as conn:
        tables = _tables(conn)
        taxonomy = _taxonomy(conn, tenant, tables)
        if (entity or entity_name) and not (tables["feedback_entities"] and tables["feedback_ner_state"]):
            return []
        if topic_id is not None and not (taxonomy and tables["sm_topic_assignments"]):
            return []
        if has_ner and not tables["feedback_ner_state"]:
            return []
        ident = taxonomy["taxonomy_id"] if taxonomy else None
        sql, extra = build_review_query(source=source,keyword=keyword,entity=entity,
                    entity_name=entity_name,topic_id=topic_id,has_ner=has_ner,
                    tables=tables,taxonomy_id=ident)
        return list(conn.execute(sql, [tenant, *extra]))


def highlights(text: str, mentions: list[dict]) -> str:
    """Render only escaped review text/NER spans. Keep nested entities in the table."""
    candidates = [m for m in mentions if type(m.get("start")) is int and type(m.get("end")) is int
                  and 0 <= m["start"] < m["end"] <= len(text)
                  and text[m["start"]:m["end"]] == m.get("text")]
    chosen = []
    for m in sorted(candidates, key=lambda x: (x["start"]-x["end"], x["start"])):
        if not any(m["start"] < x["end"] and x["start"] < m["end"] for x in chosen):
            chosen.append(m)
    parts, pos = [], 0
    for m in sorted(chosen, key=lambda x:x["start"]):
        parts.append(escape(text[pos:m["start"]]))
        score = m.get("confidence")
        tip = f'{m["label"]} · {score:.0%}' if isinstance(score,(float,int)) else str(m["label"])
        parts.append(f'<mark title="{escape(tip,quote=True)}" '
                     f'style="background:{COLORS.get(m["label"],"#e2e8f0")};color:#17202f;'
                     f'border-radius:5px;padding:2px 4px">{escape(m["text"])}'
                     f'<sup> {escape(str(m["label"]))}</sup></mark>')
        pos=m["end"]
    return '<div style="white-space:pre-wrap;line-height:2.2">' + ''.join(parts) + escape(text[pos:]) + '</div>'


def topic_match(tenant: str, text: str, device: str = "auto") -> dict | None:
    """Read stored taxonomy centroids; classify pasted feedback without saving it."""
    guard(tenant)
    from enrichment import topics
    with connect() as conn:
        tables = _tables(conn)
        taxonomy = _taxonomy(conn, tenant, tables)
        if taxonomy is None:
            return None
        model, ids, centers = topics._read_taxonomy(conn, tenant, taxonomy["taxonomy_id"])
        names = {r["topic_id"]:r["name"] for r in conn.execute("""SELECT topic_id,name
            FROM sm_topic_definitions WHERE tenant_id=%s AND taxonomy_id=%s""",
            (tenant, taxonomy["taxonomy_id"]))}
    with TOPIC_LOCK:
        sha, path = topics._snapshot(model["model_name"], model["model_revision"], True)
        if sha != model["model_revision"]:
            raise RuntimeError("Topic embedding model revision mismatch")
        encoder = topics._encoder(path, topics._device(device))
        vector = encoder.encode([text], normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        if vector.shape[1] != centers.shape[1]:
            raise RuntimeError("Topic vector dimension mismatch")
        topic_id, score = topics._choose_topics(vector, ids, centers,
                float(model["settings"]["min_similarity"]),
                float(model["settings"]["min_margin"]))[0]
    return {"name":names.get(topic_id,"Unassigned / emerging"), "score":float(score),
            "approved":taxonomy["approved"], "model":model["model_name"]}


def analyze(tenant: str, text: str, ner, model_lock: Lock, *, device: str = "auto",
            opinion_model=None) -> dict:
    """Run existing engines independently; preview does not write to the database."""
    guard(tenant)
    if not text.strip() or len(text) > MAX_CHARS:
        raise ValueError(f"Provide 1–{MAX_CHARS} characters")
    result = {"entities":[], "ner_model":ner.name, "topic":None, "opinions":None,"warnings":[]}
    try:
        with model_lock:
            result["entities"] = ner.extract(text)
    except Exception as exc:
        result["warnings"].append(f"NER unavailable: {exc}")
    if result["entities"]:
        try:
            from enrichment.ned import resolve
            with connect() as conn:
                catalog = exists(conn,"entity_catalog")
                if catalog:
                    for entity in result["entities"][:30]:
                        if entity["label"] in {"brand","product","company","organization"}:
                            entity["catalog_link"] = resolve(conn, tenant=tenant,
                                mention=entity["text"], label=entity["label"])
        except Exception as exc:
            result["warnings"].append(f"Catalog linking unavailable: {exc}")
    try:
        result["topic"] = topic_match(tenant,text,device=device)
    except Exception as exc:
        result["warnings"].append(f"Topic model unavailable: {exc}")
    if opinion_model is not None:
        try:
            from agent import analyze as extract
            qwen, lock = opinion_model
            with lock:
                result["opinions"] = extract(qwen,text).model_dump()
        except Exception as exc:
            result["warnings"].append(f"Opinion model unavailable: {exc}")
    return result
