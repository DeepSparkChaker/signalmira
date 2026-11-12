"""SignalMira local DEMO aspect discovery and saved-group analytics.

This is a Streamlit multipage entry; no model/GPU work and no review writes.
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from enrichment import aspect_groups as ag
from frontend import service
from storage.postgres import connect

st.set_page_config(page_title='SignalMira · Aspect Groups', page_icon='◈', layout='wide')
st.title('◈ Aspect Groups')
st.caption('Discover saved opinions → choose aspects → save reusable groups → explore reviews and weekly trends.')

if os.getenv('SM_ENABLE_ASPECT_LAB') != '1':
    st.info('Local preview disabled. Set SM_ENABLE_ASPECT_LAB=1 to enable.')
    st.stop()

st.warning('LOCAL DEMO ONLY · Workspace selector is not authentication. Do not expose this page publicly.')

choices = service.tenants()
if not choices:
    st.error('Configure SIGNALMIRA_DEMO_TENANTS in your local environment.')
    st.stop()

initial = st.session_state.get('_workspace')
tenant = st.selectbox('Workspace', choices, index=choices.index(initial) if initial in choices else 0)
if 'DATABASE_URL' not in os.environ:
    st.error('DATABASE_URL is missing. Load your local database environment before starting Streamlit.')
    st.stop()

try:
    with connect(os.environ['DATABASE_URL']) as conn:
        versions = ag.list_extractor_runs(conn, tenant)
        if not versions:
            st.info('No CURRENT saved opinion extractions found. Preview scripts do not persist results. Use the one-review sync command in the README.')
        else:
            labels = [f"{v['model_name']} · {v['review_count']} reviews · {v['extractor_key'][:10]}…" for v in versions]
            version_index = st.selectbox('Saved extraction model + prompt version', range(len(versions)),
                                         format_func=lambda i: labels[i])
            extractor_key = versions[version_index]['extractor_key']
            facets = ag.list_aspects(conn, tenant, extractor_key)
            aspect_options = [r['aspect'] for r in facets]
            counts = {r['aspect']: r['review_count'] for r in facets}
            if not facets:
                st.info('No opinions available under this extraction version.')
                st.stop()
            st.caption(f'{len(facets)} available aspect labels (top 500 by review count). Counts refer to distinct saved reviews.')

            groups = ag.list_groups(conn, tenant, extractor_key)
            labels = ['New group'] + [f"{g['name']} (v{g['version']})" for g in groups]
            selection = st.selectbox('Saved aspect group', range(len(labels)), format_func=lambda i: labels[i])
            group = groups[selection-1] if selection else None
            if group:
                st.caption('Membership is versioned. Editing it creates a new version; old definitions remain in PostgreSQL.')
            name = st.text_input('Group name', value=group['name'] if group else 'Battery complaints',
                                 max_chars=120, disabled=bool(group),
                                 key=f'ag_name_{tenant}_{extractor_key}_{selection}')

            # Preserve saved rare aspects even when they fall outside the 500 top labels.
            options = list(dict.fromkeys(aspect_options + (list(group['aspects']) if group else [])))
            chosen = st.multiselect('Select exact existing aspect names', options=options,
                                    default=list(group['aspects']) if group else [],
                                    format_func=lambda a: f'{a} ({counts.get(a, "saved")})',
                                    key=f'ag_choose_{tenant}_{extractor_key}_{selection}',
                                    max_selections=ag.MAX_ASPECTS)
            can_write = os.getenv('SM_ENABLE_DEMO_ASPECT_WRITES') == '1'
            left, right = st.columns([1, 3])
            with left:
                if st.button('Save group' if group is None else 'Save new version',
                             disabled=not chosen or not can_write, type='primary'):
                    try:
                        if group is None:
                            ag.create_group(conn, tenant, extractor_key, name, chosen)
                        else:
                            ag.update_group(conn, tenant, str(group['group_id']),
                                            group['version'], chosen)
                        st.success('Saved group definition in PostgreSQL.')
                        st.rerun()
                    except Exception as exc:
                        st.error(f'Could not save: {type(exc).__name__}: {exc}')
            with right:
                if not can_write:
                    st.caption('Saving is off. To allow LOCAL demo group writes: export SM_ENABLE_DEMO_ASPECT_WRITES=1')

            st.divider()
            st.subheader('Explore selected feedback')
            f1, f2, f3 = st.columns(3)
            source = f1.text_input('Source (optional)', placeholder='uci-amazon', max_chars=256)
            product = f2.text_input('Product ID (optional)', max_chars=256)
            only_negative = f3.checkbox('Only reviews with negative selected aspects', value=True)
            date_filter = st.checkbox('Filter by original review date', value=False)
            begin = end = None
            if date_filter:
                a, b = st.columns(2)
                begin = a.date_input('From', value=date(2025, 1, 1))
                end = b.date_input('To', value=date.today())
            selected = chosen or (list(group['aspects']) if group else [])
            if st.button('Analyze selected aspects', disabled=not selected):
                try:
                    kw = dict(source=source.strip() or None, product_id=product.strip() or None,
                              date_from=begin, date_to=end)
                    weekly = ag.weekly_trend(conn, tenant, extractor_key, selected, **kw)
                    rows = ag.find_reviews(conn, tenant, extractor_key, selected,
                                           negative_only=only_negative, limit=30, **kw)
                    st.session_state['ag_result'] = dict(
                        tenant=tenant, key=extractor_key, selected=selected, weekly=weekly, rows=rows)
                except Exception as exc:
                    st.error(f'Analysis failed: {type(exc).__name__}: {exc}')
            result = st.session_state.get('ag_result')
            if result and result['tenant'] == tenant and result['key'] == extractor_key:
                st.caption('Result of last analysis. Press Analyze again after changing selection/filters.')
                weekly = result['weekly']
                st.subheader('Weekly trends (UTC, dated reviews only)')
                st.caption('Negative % = distinct reviews with ≥1 negative chosen aspect / distinct reviews mentioning any chosen aspect. Product/date coverage affects interpretation.')
                if weekly:
                    st.line_chart([{'week': r['week'], 'negative_pct': float(r['negative_pct'])} for r in weekly],
                                  x='week', y='negative_pct')
                    st.dataframe(weekly, hide_index=True, use_container_width=True)
                else:
                    st.info('No matching reviews with review_date. Undated reviews can still appear below.')
                st.subheader('Matching saved reviews (maximum 30 distinct reviews)')
                rows = result['rows']
                if rows:
                    st.dataframe(rows, hide_index=True, use_container_width=True)
                else:
                    st.info('No saved reviews match these filters.')

        st.divider()
        st.subheader('Test one review already saved in PostgreSQL')
        a, b = st.columns(2)
        sample_source = a.text_input('Saved source', value='uci-amazon', max_chars=256)
        sample_id = b.text_input('Saved review ID', value='uci-amazon-0694', max_chars=256)
        if st.button('Inspect stored review'):
            try:
                # If no model version is selected, show all currently saved opinion runs.
                review = ag.saved_review(conn, tenant, sample_source, sample_id,
                                         extractor_key=extractor_key if versions else None)
                if review is None:
                    st.warning('Review not found in this tenant. Confirm the source and ID.')
                else:
                    st.write('**Original saved review:**', review['text'])
                    st.caption(f"Source: {review['source']} · review_id: {review['review_id']} · date: {review['review_date']}")
                    if review['opinions']:
                        st.dataframe(review['opinions'], hide_index=True, use_container_width=True)
                    else:
                        st.warning('This review has no CURRENT saved opinions for the chosen extraction version. Run the one-review sync in the README.')
            except Exception as exc:
                st.error(f'Could not inspect review: {type(exc).__name__}: {exc}')
except Exception as exc:
    st.error(f'Database/feature unavailable: {type(exc).__name__}: {exc}')
    st.caption('Apply migrations/020_sentiment_opinions.sql and migrations/021_aspect_groups.sql; verify DATABASE_URL and Docker.')
