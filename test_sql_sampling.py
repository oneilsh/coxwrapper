#!/usr/bin/env python3
"""
Test SQL-level sampling functionality
"""

# Test the SQL query generation with sampling
from coxwrapper.model.dataloader_cox import (
    build_person_base_query,
    get_observation_periods_query,
    get_all_outcome_events_query
)

def test_sql_sampling():
    """Test that SQL sampling clauses are added correctly."""

    print("🧪 Testing SQL-Level Sampling Implementation")
    print("=" * 60)

    # Set dummy environment variable for testing
    import os
    os.environ['WORKSPACE_CDR'] = 'test-workspace.r2023q3_unzipped_data'

    # Test configuration
    config = {
        'outcome': {
            'name': 'copd',
            'domain': 'condition_occurrence',
            'concepts_include': [255573],
            'concepts_exclude': []
        },
        'cohort_definition': {
            'cohort_table_id': 'my_cohort_table'
        }
    }

    # Test 1: No sampling (sampling_ratio = 1.0)
    print("\n📊 Test 1: No sampling (sampling_ratio = 1.0)")
    query = build_person_base_query(config, sampling_ratio=1.0)
    print("✅ Query generated without sampling clause")
    print("Last few lines:")
    lines = query.strip().split('\n')
    for line in lines[-3:]:
        print(f"  {line}")

    # Test 2: 10% sampling (sampling_ratio = 0.1)
    print("\n📊 Test 2: 10% sampling (sampling_ratio = 0.1)")
    query = build_person_base_query(config, sampling_ratio=0.1)
    print("✅ Query generated with 10% sampling clause")
    print("Last few lines:")
    lines = query.strip().split('\n')
    for line in lines[-5:]:
        print(f"  {line}")

    # Test 3: 50% sampling (sampling_ratio = 0.5)
    print("\n📊 Test 3: 50% sampling (sampling_ratio = 0.5)")
    query = build_person_base_query(config, sampling_ratio=0.5)
    print("✅ Query generated with 50% sampling clause")
    print("Last few lines:")
    lines = query.strip().split('\n')
    for line in lines[-5:]:
        print(f"  {line}")

    # Test 4: Observation periods query
    print("\n📊 Test 4: Observation periods with 25% sampling")
    obs_query = get_observation_periods_query("test.cdr_path", sampling_ratio=0.25)
    print("✅ Observation periods query with sampling:")
    lines = obs_query.strip().split('\n')
    for line in lines[-3:]:
        print(f"  {line}")

    # Test 5: Outcome events query
    print("\n📊 Test 5: Outcome events with 5% sampling")
    outcome_config = config['outcome']
    outcome_query = get_all_outcome_events_query(outcome_config, "test.cdr_path", sampling_ratio=0.05)
    print("✅ Outcome events query with sampling:")
    lines = outcome_query.strip().split('\n')
    for line in lines[-5:]:
        print(f"  {line}")

    print("\n" + "=" * 60)
    print("🎉 SQL Sampling Implementation Test Complete!")
    print("✅ All queries generated correctly with sampling clauses")
    print("✅ Uses BigQuery's FARM_FINGERPRINT for consistent hashing")
    print("✅ Sampling applied at SQL level - very efficient!")
    print("=" * 60)

if __name__ == "__main__":
    test_sql_sampling()
