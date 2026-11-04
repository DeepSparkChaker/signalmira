"""Dependency-free tests for discovery decisions and historical quotas."""
from datetime import datetime, timedelta, timezone
import random

import pytest
from enrichment.topic_policy import Policy, allocate_strata, decide

NOW = datetime(2026, 10, 31, tzinfo=timezone.utc)


def decide_for(new=300, days=8, assigned=300, unassigned=0, pending=False, active=True, p=None):
    return decide(p or Policy(), new_count=new, assigned_count=assigned,
                  unassigned_count=unassigned, pending=pending, active=active,
                  last_discovery_at=None if days is None else NOW-timedelta(days=days), now=NOW)


def test_weekly():
    r=decide_for()
    assert r['due'] and 'weekly' in r['reasons']


def test_cooldown():
    r=decide_for(new=6000,days=.5)
    assert not r['due']


def test_volume():
    assert 'volume' in decide_for(new=5000,days=2)['reasons']


def test_no_new():
    assert not decide_for(new=0)['due']


def test_pending_candidate():
    assert not decide_for(new=99999,pending=True)['due']


def test_bootstrap_minimum():
    assert decide_for(new=100,days=None,active=False)['due']
    assert not decide_for(new=99,days=None,active=False)['due']


def test_drift_assigned_coverage():
    assert 'emerging_topics' in decide_for(new=600,days=2,assigned=600,unassigned=210)['reasons']
    assert 'emerging_topics' not in decide_for(new=600,days=2,assigned=200,unassigned=200)['reasons']


def test_drift_not_without_active_taxonomy():
    assert 'emerging_topics' not in decide_for(new=600,days=2,assigned=600,unassigned=210,active=False)['reasons']


def test_weekly_below_minimum():
    assert not decide_for(new=199,days=50)['due']


@pytest.mark.parametrize('changes',[
    dict(sample_mode='nonsense'),dict(new_fraction=1.01),
    dict(days=0),dict(max_docs=1),dict(profile='invalid'),
])
def test_invalid_policy(changes):
    with pytest.raises(ValueError):
        Policy(**changes)


def test_strata_quota_under_capacity():
    sizes={(-1,'en'):1000,(1,'en'):300,(2,'zh'):5,(3,'fr'):9}
    q=allocate_strata(sizes,200)
    assert sum(q.values())==200
    assert all(0<n<=sizes[k] for k,n in q.items())
    assert q[(2,'zh')]>0


def test_strata_all_selected_when_room():
    sizes={(1,'en'):3,(2,'fr'):4}
    assert allocate_strata(sizes,100)==sizes


def test_strata_reproducible():
    sizes={(i%7,['en','fr','es'][i%3]):random.Random(i).randint(1,100) for i in range(30)}
    assert allocate_strata(sizes,100)==allocate_strata(sizes,100)


def test_strata_zero_seats():
    assert allocate_strata({(0,'en'):2},0)=={}
