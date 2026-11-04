"""Optional PostgreSQL integration test. Never runs without TEST_DATABASE_URL."""
import hashlib
import json
import os
from uuid import uuid4

import pytest

from enrichment import topic_policy as policy
from enrichment import topics


@pytest.mark.skipif(not os.getenv('TEST_DATABASE_URL'), reason='Set TEST_DATABASE_URL to test live PostgreSQL')
def test_sample_and_version_guard():
    psycopg = pytest.importorskip('psycopg')
    from psycopg.rows import dict_row
    tenant = 'sm_test_' + uuid4().hex[:16]
    with psycopg.connect(os.environ['TEST_DATABASE_URL'],autocommit=True,row_factory=dict_row) as conn:
        topics.initialize(conn)
        try:
            records=[]
            for i in range(60):
                text=f'Synthetic battery complaint number {i}'
                records.append((tenant,'test-upload',f'r{i:03}',text,
                    hashlib.sha256(text.encode()).hexdigest(),
                    json.dumps({'language':'fr' if i%3==0 else 'en'})))
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.executemany('''INSERT INTO feedback_reviews
                      (tenant_id,source,review_id,text,content_hash,attributes)
                      VALUES (%s,%s,%s,%s,%s,%s::jsonb)''',records)
                    cur.executemany('''INSERT INTO sm_topic_discovery_seen
                      (tenant_id,source,review_id,content_hash)
                      VALUES (%s,%s,%s,%s)''', [(t,s,r,h) for t,s,r,_,h,_ in records[:30]])
            policy.create_source_snapshot(conn,tenant)
            selected,stats=policy.select_sample(conn,tenant,max_docs=40,
                new_fraction=.5,sample_mode='hybrid',seed=42)
            assert len(selected)==40
            assert stats['available_new']==30 and stats['available_historical']==30
            assert stats['selected_new']==20 and stats['selected_historical']==20
            assert len({(r['source'],r['review_id']) for r in selected})==40
            changed='Changed after snapshot'
            conn.execute('''UPDATE feedback_reviews SET text=%s,content_hash=%s,updated_at=now()
                WHERE tenant_id=%s AND review_id=%s''',
                (changed,hashlib.sha256(changed.encode()).hexdigest(),tenant,'r000'))
            with conn.transaction():
                policy.complete_discovery(conn,tenant,uuid4(),selected_rows=selected)
            # An update committed during discovery remains eligible next time.
            check=policy.check(conn,tenant)
            assert check['new_or_changed']==11  # 10 unsampled new + 1 changed after snapshot
        finally:
            conn.execute('DELETE FROM sm_topic_discovery_state WHERE tenant_id=%s',(tenant,))
            conn.execute('DELETE FROM sm_topic_policies WHERE tenant_id=%s',(tenant,))
            conn.execute('DELETE FROM feedback_reviews WHERE tenant_id=%s',(tenant,))
