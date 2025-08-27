import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import yaml
import os
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mock environment variables for local testing if not in AoU Workbench
# os.environ["WORKSPACE_CDR"] = "all-of-us-research-workbench-####.r2023q3_unzipped_data"

def load_configuration(config_filepath: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(config_filepath, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_filepath}")
        raise RuntimeError(f"Configuration file not found: {config_filepath}")
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration YAML: {e}")
        raise RuntimeError(f"Error parsing configuration YAML: {e}")

def get_aou_cdr_path() -> str:
    """Returns the base path for the All of Us Controlled Tier Dataset."""
    if "WORKSPACE_CDR" not in os.environ:
        raise EnvironmentError("WORKSPACE_CDR environment variable not set. "
                               "This is required in the All of Us Workbench.")
    return os.environ["WORKSPACE_CDR"]

def _build_ancestor_descendant_sql(cdr_path: str, ancestor_concept_ids: List[int]) -> str:
    """
    Builds a SQL subquery to select all descendant concept IDs for given ancestor concept IDs
    using cb_criteria_ancestor and cb_criteria.
    This pattern is adapted from the user's provided SQL snippet, assuming is_standard=1 and is_selectable=1.
    """
    if not ancestor_concept_ids:
        # Return a subquery that yields no concepts if no ancestors provided (e.g., for empty include)
        return "SELECT concept_id FROM `dummy_table` WHERE FALSE" # BigQuery requires a FROM clause

    ancestor_ids_str = ','.join(map(str, ancestor_concept_ids))

    sql = f"""
    SELECT
        DISTINCT ca.descendant_id
    FROM
        `{cdr_path}.cb_criteria_ancestor` ca
    JOIN
        (
            SELECT
                DISTINCT c.concept_id
            FROM
                `{cdr_path}.cb_criteria` c
            JOIN
                (
                    SELECT
                        cast(cr.id as string) as id
                    FROM
                        `{cdr_path}.cb_criteria` cr
                    WHERE
                        concept_id IN ({ancestor_ids_str})
                        AND full_text LIKE '%_rank1]%'
                ) a
                ON (
                    c.path LIKE CONCAT('%.', a.id, '.%')
                    OR c.path LIKE CONCAT('%.', a.id)
                    OR c.path LIKE CONCAT(a.id, '.%')
                    OR c.path = a.id
                )
            WHERE
                is_standard = 1
                AND is_selectable = 1
        ) b
        ON (ca.ancestor_id = b.concept_id)
    """
    return sql

def _get_concept_filter_sql(domain_table_alias: str, concept_id_col_name: str, concepts_config: Dict[str, Any], cdr_path: str) -> str:
    """
    Generates SQL filter condition for including/excluding concepts, supporting ancestor mapping.
    """
    include_concepts = concepts_config.get('concepts_include', [])
    exclude_concepts = concepts_config.get('concepts_exclude', [])
    map_to_descendants = concepts_config.get('map_to_descendants', False)

    include_conditions = []
    exclude_conditions = []

    if include_concepts:
        if map_to_descendants:
            include_sql = _build_ancestor_descendant_sql(cdr_path, include_concepts)
            include_conditions.append(f"{domain_table_alias}.{concept_id_col_name} IN ({include_sql})")
        else:
            include_conditions.append(f"{domain_table_alias}.{concept_id_col_name} IN ({','.join(map(str, include_concepts))})")

    if exclude_concepts:
        # Exclude descendants if ancestors are provided for exclusion, otherwise direct exclude
        if map_to_descendants: # Assuming if you map_to_descendants for include, you'd for exclude too. Or add separate flag.
            exclude_sql = _build_ancestor_descendant_sql(cdr_path, exclude_concepts)
            exclude_conditions.append(f"{domain_table_alias}.{concept_id_col_name} NOT IN ({exclude_sql})")
        else:
            exclude_conditions.append(f"{domain_table_alias}.{concept_id_col_name} NOT IN ({','.join(map(str, exclude_concepts))})")
    
    final_conditions = []
    if include_conditions:
        final_conditions.append(f"({' OR '.join(include_conditions)})") # Using OR if multiple include statements were generated
    if exclude_conditions:
        final_conditions.append(f"({' AND '.join(exclude_conditions)})")

    return " AND ".join(final_conditions) if final_conditions else ""


# --- MODIFIED: build_domain_query to be more flexible and return raw events ---
# This function will now focus on pulling raw events within a person's record
# and then we'll apply time window filtering in load_data_from_bigquery.
def build_domain_events_query(domain_name: str, concept_config: Dict[str, Any], cdr_path: str) -> str:
    """
    Builds a SQL query to extract all relevant events for a given domain and concept configuration
    (e.g., all COPD diagnoses, all BMI measurements).
    Returns basic columns needed for time-based filtering later.
    """
    domain_table_name = domain_name
    domain_table = f"`{cdr_path}.{domain_table_name}`"
    concept_id_col_name = ""
    date_col_name = ""
    value_col_name = "NULL"
    value_type = "" # 'number', 'concept', 'string'

    if domain_name == 'condition_occurrence':
        concept_id_col_name = 'condition_concept_id'
        date_col_name = 'condition_start_datetime'
    elif domain_name == 'observation':
        concept_id_col_name = 'observation_concept_id'
        date_col_name = 'observation_datetime'
        value_col_name = 'value_as_concept_id' # For categorical answers
        value_type = 'concept'
        # Special handling for survey questions in observation table for All of Us
        if concept_config.get('domain') == 'ds_survey': # This implies it's a survey feature
            concept_id_col_name = 'observation_concept_id' # This is the question concept ID
            date_col_name = 'observation_datetime' # Date of the survey response
            value_col_name = 'value_as_concept_id' # The answer concept ID
            value_type = 'concept'
    elif domain_name == 'measurement':
        concept_id_col_name = 'measurement_concept_id'
        date_col_name = 'measurement_datetime'
        value_col_name = 'value_as_number'
        value_type = 'number'
    elif domain_name == 'drug_exposure':
        concept_id_col_name = 'drug_concept_id'
        date_col_name = 'drug_exposure_start_datetime'
    elif domain_name == 'procedure_occurrence':
        concept_id_col_name = 'procedure_concept_id'
        date_col_name = 'procedure_datetime'
    else:
        logging.warning(f"Domain '{domain_name}' not fully supported for event extraction. Skipping.")
        return "" # Return empty if domain not handled

    concept_filter_conditions = _get_concept_filter_sql("t", concept_id_col_name, concept_config, cdr_path)

    # Collect all WHERE conditions
    where_clauses = [f"t.{date_col_name} IS NOT NULL"]
    if concept_filter_conditions:
        where_clauses.append(concept_filter_conditions)

    final_where_clause = ""
    if where_clauses:
        final_where_clause = f"WHERE {' AND '.join(where_clauses)}"

    sql = f"""
    SELECT
        t.person_id,
        t.{concept_id_col_name} AS concept_id,
        t.{date_col_name} AS event_datetime,
        {value_col_name} AS value, -- Include value for measurements/observations
        '{concept_config['name']}' AS feature_name,
        '{concept_config['domain']}' AS domain_name,
        '{value_type}' AS value_type
    FROM
        {domain_table} t
    {final_where_clause}
    """
    return sql

# --- NEW: Helper to get all observation periods ---
def get_observation_periods_query(cdr_path: str) -> str:
    return f"""
    SELECT
        person_id,
        observation_period_start_date,
        observation_period_end_date
    FROM
        `{cdr_path}.observation_period`
    WHERE
        observation_period_start_date IS NOT NULL AND observation_period_end_date IS NOT NULL
        AND DATE_DIFF(observation_period_end_date, observation_period_start_date, DAY) >= 0
    """

# --- NEW: Helper to get all outcome events ---
def get_all_outcome_events_query(outcome_config: Dict[str, Any], cdr_path: str) -> str:
    outcome_domain = outcome_config['domain']
    outcome_concept_id_col_name = 'condition_concept_id' # Assuming condition_occurrence for outcome
    outcome_date_col_name = 'condition_start_datetime'

    concept_filter_conditions = _get_concept_filter_sql("t", outcome_concept_id_col_name, outcome_config, cdr_path)

    # Collect all WHERE conditions
    where_clauses = [f"t.{outcome_date_col_name} IS NOT NULL"]
    if concept_filter_conditions:
        where_clauses.append(concept_filter_conditions)

    final_where_clause = ""
    if where_clauses:
        final_where_clause = f"WHERE {' AND '.join(where_clauses)}"

    return f"""
    SELECT
        t.person_id,
        t.{outcome_date_col_name} AS outcome_datetime
    FROM
        `{cdr_path}.{outcome_domain}` t
    {final_where_clause}
    """


def load_data_from_bigquery(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Loads data from Google BigQuery based on the provided configuration.
    Uses pd.read_gbq() compatible with AoU environment authentication.
    Implements cohort construction, random time_0 selection, and time-to-event outcome derivation.
    """
    cdr_path = get_aou_cdr_path()

    logging.info("Step 1: Fetching base person demographic data.")
    base_person_query = build_person_base_query(config)
    person_df = pd.read_gbq(base_person_query)
    logging.info(f"Base person data loaded. Shape: {person_df.shape}")

    if person_df.empty:
        logging.warning("No persons found matching base cohort criteria. Returning empty DataFrame.")
        return pd.DataFrame()

    # Early sampling for development/testing (before expensive BigQuery operations)
    max_patients = config.get('max_patients', None)
    if max_patients and len(person_df) > max_patients:
        logging.info(f"Early sampling: Taking {max_patients} patients from {len(person_df)} total (for development/testing)")
        person_df = person_df.sample(n=max_patients, random_state=42).reset_index(drop=True)

    logging.info("Step 2: Fetching observation periods for all persons.")
    obs_period_query = get_observation_periods_query(cdr_path)
    obs_period_df = pd.read_gbq(obs_period_query)
    logging.info(f"Observation periods loaded. Shape: {obs_period_df.shape}")

    logging.info(f"Step 3: Fetching all outcome events (COPD) for potential filtering.")
    outcome_config = config.get('outcome')
    if not outcome_config or 'domain' not in outcome_config:
        raise ValueError("Outcome configuration missing or incomplete in YAML.")

    all_outcome_events_query = get_all_outcome_events_query(outcome_config, cdr_path)
    all_outcome_events_df = pd.read_gbq(all_outcome_events_query)
    logging.info(f"All outcome events loaded. Shape: {all_outcome_events_df.shape}")

    # --- Step 4: Determine a single random time_0 for each person ---
    # This involves:
    #   a) Filtering observation periods by minimum lookback/follow-up requirements.
    #   b) Excluding time_0 candidates that are after an outcome event (prevalent cases).
    #   c) Randomly selecting one valid time_0 per person.

    # Default minimum lookback (e.g., 1 year) and follow-up (e.g., 5 years) for time_0 selection
    MIN_LOOKBACK_DAYS = 365
    MIN_FOLLOWUP_DAYS = 365 * 5
    
    logging.info("Step 4: Determining random time_0 for each person...")
    # Join observation periods with outcome events to filter valid time_0 ranges
    # Use left join to include all observation periods, even those without outcomes
    logging.info("Merging observation periods with outcome events...")
    logging.info(f"  Observation periods: {len(obs_period_df)} rows")
    logging.info(f"  Outcome events: {len(all_outcome_events_df)} rows")

    person_obs_outcome = pd.merge(obs_period_df, all_outcome_events_df, on='person_id', how='left')
    logging.info(f"  After merge: {len(person_obs_outcome)} rows")

    # Vectorized date operations for better performance
    logging.info("Performing vectorized time_0 calculations...")

    # Convert dates to datetime once
    person_obs_outcome['obs_start'] = pd.to_datetime(person_obs_outcome['observation_period_start_date'])
    person_obs_outcome['obs_end'] = pd.to_datetime(person_obs_outcome['observation_period_end_date'])
    person_obs_outcome['outcome_date'] = pd.to_datetime(person_obs_outcome['outcome_datetime'])

    today = datetime.today().date()

    # Calculate earliest and latest possible time_0 for each person
    person_obs_outcome['earliest_time_0'] = person_obs_outcome['obs_start'] + pd.Timedelta(days=MIN_LOOKBACK_DAYS)
    person_obs_outcome['latest_time_0_raw'] = person_obs_outcome['obs_end'] - pd.Timedelta(days=MIN_FOLLOWUP_DAYS)

    # Ensure time_0 is not in the future
    person_obs_outcome['latest_time_0'] = person_obs_outcome['latest_time_0_raw'].dt.date.clip(upper=today)

    # If there's an outcome, time_0 must be before the outcome date
    mask_has_outcome = person_obs_outcome['outcome_date'].notna()

    # For patients with outcomes, take the minimum of latest_time_0 and outcome_date, then subtract 1 day
    for idx in person_obs_outcome[mask_has_outcome].index:
        latest_time = person_obs_outcome.loc[idx, 'latest_time_0']
        outcome_time = person_obs_outcome.loc[idx, 'outcome_date'].date()
        person_obs_outcome.loc[idx, 'latest_time_0'] = min(latest_time, outcome_time - timedelta(days=1))

    # Filter valid time windows
    valid_window_mask = person_obs_outcome['earliest_time_0'].dt.date <= person_obs_outcome['latest_time_0']
    person_obs_outcome = person_obs_outcome[valid_window_mask]
    logging.info(f"  Valid time windows: {len(person_obs_outcome)} patients ({len(person_obs_outcome)/len(obs_period_df)*100:.1f}% of observation periods)")

    if person_obs_outcome.empty:
        logging.warning("No persons with valid time windows found.")
        return pd.DataFrame()

    # Calculate time window sizes for random sampling
    person_obs_outcome['window_days'] = (person_obs_outcome['latest_time_0'] - person_obs_outcome['earliest_time_0'].dt.date).dt.days + 1
    person_obs_outcome = person_obs_outcome[person_obs_outcome['window_days'] > 0]

    if person_obs_outcome.empty:
        logging.warning("No persons with positive time windows found.")
        return pd.DataFrame()

    # Generate random offsets within valid windows
    np.random.seed(42)  # For reproducibility
    person_obs_outcome['random_offset'] = person_obs_outcome['window_days'].apply(lambda days: np.random.randint(0, days))
    person_obs_outcome['selected_time_0'] = person_obs_outcome['earliest_time_0'] + pd.to_timedelta(person_obs_outcome['random_offset'], unit='D')

    # Create time_0 DataFrame efficiently
    time_0_df = person_obs_outcome[[
        'person_id',
        'selected_time_0',
        'obs_start',
        'obs_end',
        'outcome_date'
    ]].rename(columns={
        'selected_time_0': 'time_0',
        'obs_start': 'observation_period_start_date',
        'obs_end': 'observation_period_end_date',
        'outcome_date': 'actual_outcome_datetime'
    })

    # Convert to ISO format strings for BigQuery compatibility
    time_0_df['time_0'] = time_0_df['time_0'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    time_0_df['observation_period_start_date'] = time_0_df['observation_period_start_date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    time_0_df['observation_period_end_date'] = time_0_df['observation_period_end_date'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    # Handle NaN values in outcome datetime
    time_0_df['actual_outcome_datetime'] = time_0_df['actual_outcome_datetime'].apply(
        lambda x: x.strftime('%Y-%m-%dT%H:%M:%S') if pd.notna(x) else None
    )
    
    # Select one time_0 per person if multiple observation periods/outcomes resulted in multiple candidates
    # This just picks one if there are overlaps, the prior logic ensures it's valid per person
    time_0_df = time_0_df.groupby('person_id').sample(n=1, random_state=42).reset_index(drop=True)
    
    logging.info(f"Determined time_0 for {time_0_df.shape[0]} unique persons.")

    # Merge time_0 back into the base person_df
    person_df = pd.merge(person_df, time_0_df[['person_id', 'time_0', 'observation_period_start_date', 'observation_period_end_date', 'actual_outcome_datetime']], on='person_id', how='inner')

    if person_df.empty:
        logging.warning("No persons with valid time_0 found after filtering. Returning empty DataFrame.")
        return pd.DataFrame()

    # Optional: Add sampling for development/testing with large datasets
    # This can be controlled via configuration
    max_patients = config.get('max_patients', None)
    if max_patients and len(person_df) > max_patients:
        logging.info(f"Sampling {max_patients} patients from {len(person_df)} total (for development/testing)")
        person_df = person_df.sample(n=max_patients, random_state=42).reset_index(drop=True)


    # --- Step 5: Derive Outcome (time_to_event_days, event_observed) ---
    logging.info("Step 5: Deriving time_to_event_days and event_observed...")
    # Convert time_0, obs_end, and actual_outcome to datetime objects and make them timezone-naive
    # person_df['time_0_dt'] = pd.to_datetime(person_df['time_0']).dt.tz_localize(None) # Make timezone-naive
    # person_df['obs_end_dt'] = pd.to_datetime(person_df['observation_period_end_date']).dt.tz_localize(None) # Make timezone-naive
    # person_df['actual_outcome_dt'] = pd.to_datetime(person_df['actual_outcome_datetime']).dt.tz_localize(None) # Make timezone-naive
    common_datetime_format = "%Y-%m-%dT%H:%M:%S.%f%z"

    # Now, use the format argument in each call to pd.to_datetime
    person_df['time_0_dt'] = pd.to_datetime(person_df['time_0'], format=common_datetime_format, errors='coerce').dt.tz_localize(None)
    person_df['obs_end_dt'] = pd.to_datetime(person_df['observation_period_end_date'], format=common_datetime_format, errors='coerce').dt.tz_localize(None)
    person_df['actual_outcome_dt'] = pd.to_datetime(person_df['actual_outcome_datetime'], format=common_datetime_format, errors='coerce').dt.tz_localize(None)


    # Calculate event_observed (1 if outcome occurred AFTER time_0, 0 otherwise)
    # The 'actual_outcome_dt' here is the first outcome overall, which we know is >= time_0 due to filtering
    person_df['event_observed'] = person_df['actual_outcome_dt'].notna().astype(int)


    # Calculate time_to_event (duration from time_0 to event or censoring)
    # If event occurred: time_to_event = outcome_dt - time_0_dt
    # If censored: time_to_event = obs_end_dt - time_0_dt
    
    # Ensure time_to_event is at least 1 day (or a small positive number) for lifelines
    time_to_event_raw = np.where(
    person_df['event_observed'] == 1,
    (person_df['actual_outcome_dt'] - person_df['time_0_dt']).dt.days,
    (person_df['obs_end_dt'] - person_df['time_0_dt']).dt.days
    )
    person_df['time_to_event_days'] = np.maximum(time_to_event_raw, 1).astype(float)
    logging.info("Time-to-event and event_observed derived.")
    logging.info(f"Derived outcomes: min_time={person_df['time_to_event_days'].min()}, max_time={person_df['time_to_event_days'].max()}, events={person_df['event_observed'].sum()}")

    # --- Step 6: Fetch and Join Features based on time_0 and lookbacks ---
    logging.info("Step 6: Fetching and joining features based on time_0 and lookback windows...")
    final_df = person_df.copy() # Start with the person data and derived outcomes

    # Iterate through co_indicators and features from config
    features_to_extract = config.get('co_indicators', []) + config.get('features', [])
    
    # Filter out features that are handled differently or are outcome-related
    # (e.g., 'condition_start_datetimes', 'condition_end_datetimes' as features, not as outcome derivation)
    # The current model.py already has a strong exclusion list, which is good.
    excluded_from_feature_extraction = ['condition_duration', 'condition_start_datetimes', 'condition_end_datetimes', 'year_of_birth'] 
    # 'year_of_birth' is already in base_person_query, handled there.
    # The others are derived in model.py or are outcome-related as discussed.

    for feature_config in features_to_extract:
        feature_name = feature_config['name']
        feature_domain = feature_config['domain']

        if feature_name in excluded_from_feature_extraction or feature_domain == 'person': # Person features already in base query
            continue

        logging.info(f"Extracting feature: {feature_name} from domain: {feature_domain}")
        # Build query to get all events for this feature
        raw_events_query = build_domain_events_query(feature_domain, feature_config, cdr_path)
        if not raw_events_query: # Skip if build_domain_events_query returned empty (unsupported domain)
            continue

        feature_events_df = pd.read_gbq(raw_events_query)
        logging.info(f"Raw events for {feature_name} loaded. Shape: {feature_events_df.shape}")

        if feature_events_df.empty:
            final_df[feature_name] = np.nan # Add column of NaNs if no data
            continue

        # Convert event_datetime to datetime objects
        feature_events_df['event_datetime'] = pd.to_datetime(feature_events_df['event_datetime'])

        # --- Apply Lookback Logic & Consolidate per person_id ---
        # Merge feature events with the time_0 and observation periods
        merged_events_df = pd.merge(feature_events_df, person_df[['person_id', 'time_0_dt', 'observation_period_start_date', 'observation_period_end_date']], on='person_id', how='inner')

        feature_values = []
        for pid, group in merged_events_df.groupby('person_id'):
            person_time_0 = group['time_0_dt'].iloc[0]
            obs_start = pd.to_datetime(group['observation_period_start_date'].iloc[0])
            obs_end = pd.to_datetime(group['observation_period_end_date'].iloc[0])

            # Filter events to be within the allowed lookback window relative to time_0
            # Define lookback based on feature type (from config.yaml for now)
            # This is where configurable lookback logic gets applied per feature
            lookback_days = 365 # Default to 1 year
            is_chronic = False # Default

            # Example: Based on variable name for simplicity here, ideally from config.yaml directly
            if feature_name in ['cardiovascular_disease', 'diabetes', 'obesity', 'alcohol_use_cond']:
                is_chronic = True # These use the 'chronic/ongoing' logic
            elif feature_name in ['smoking_status_obs', 'smoking_status_measurement', 'smoking_status_survey',
                                  'alcohol_use_survey', 'alcohol_use_measurement']:
                lookback_days = 365 # Recent 1 year
            elif feature_name == 'bmi':
                lookback_days = 365 # Most recent within 1 year

            # Filter events within the lookback window
            if is_chronic:
                # For chronic: any record prior to time_0 and not explicitly ended by time_0
                # This needs condition_era table for robust implementation for chronic conditions.
                # For simplicity here, we'll consider any occurrence before time_0, assuming it's ongoing
                # if no explicit end date is present. This is a simplification.
                filtered_events = group[group['event_datetime'] <= person_time_0]
            else:
                # For acute/recent: events within specific lookback window
                lookback_start_date = person_time_0 - timedelta(days=lookback_days)
                filtered_events = group[(group['event_datetime'] >= lookback_start_date) & 
                                        (group['event_datetime'] <= person_time_0)]
            
            # --- Consolidate events into a single feature value per person ---
            feature_val = np.nan
            if not filtered_events.empty:
                if feature_config.get('type') == 'categorical':
                    # Get the most recent categorical value
                    most_recent_event = filtered_events.loc[filtered_events['event_datetime'].idxmax()]
                    feature_val = most_recent_event['concept_id'] # Use concept_id for categorical features initially
                elif feature_config.get('type') == 'binary':
                    feature_val = 1 # Presence of any event makes it 1
                elif feature_config.get('type') == 'continuous' and feature_domain == 'measurement':
                    # Get the most recent numerical value for measurements like BMI
                    most_recent_event = filtered_events.loc[filtered_events['event_datetime'].idxmax()]
                    feature_val = most_recent_event['value']
                    # Apply simple outlier handling for BMI if needed (e.g., clamp)
                    if feature_name == 'bmi':
                        feature_val = np.clip(feature_val, 10.0, 60.0) # Example clamping
                else:
                    # Default for other types, e.g., binary presence
                    feature_val = 1
            
            feature_values.append({'person_id': pid, feature_name: feature_val})
        
        if feature_values:
            feature_df_per_person = pd.DataFrame(feature_values)
            final_df = pd.merge(final_df, feature_df_per_person, on='person_id', how='left')
        else:
            final_df[feature_name] = np.nan # If no data found for any person for this feature

    logging.info(f"Final data shape after feature joining: {final_df.shape}")
    
    # Clean up temporary datetime columns
    final_df = final_df.drop(columns=['time_0_dt', 'obs_end_dt', 'actual_outcome_dt',
                                      'observation_period_start_date', 'observation_period_end_date', 'actual_outcome_datetime'], errors='ignore')
    
    # Rename 'current_age' to 'age_at_time_0' to reflect its meaning
    final_df = final_df.rename(columns={'current_age': 'age_at_time_0'})

    return final_df

# --- build_person_base_query (remains mostly same, but ensures necessary cols) ---
def build_person_base_query(config: Dict[str, Any]) -> str:
    """
    Builds the base SQL query for the person table, incorporating cohort definition
    from the configuration.
    """
    cdr_path = get_aou_cdr_path()
    
    select_clauses = [
        "person.person_id",
        # We will calculate age relative to time_0 later in Python for consistency
        "person.birth_datetime", # Keep birth_datetime to calculate age at time_0
        "person.gender_concept_id",
        "p_gender_concept.concept_name as gender",
        "person.race_concept_id",
        "p_race_concept.concept_name as race",
        "person.ethnicity_concept_id",
        "p_ethnicity_concept.concept_name as ethnicity",
        "person.sex_at_birth_concept_id",
        "p_sex_at_birth_concept.concept_name as sex_at_birth",
        "cb_search_person.age_at_consent as age_at_consent", # Keep for info, not primary feature
        "cb_search_person.has_ehr_data as has_ehr_data",
        "CASE WHEN p_observation.observation_concept_id = 1586099 AND p_observation.value_source_value = 'ConsentPermission_Yes' THEN 'Yes' ELSE 'No' END AS ehr_consent",
        "person.year_of_birth"
    ]

    from_join_clauses = [
        f"FROM `{cdr_path}.person` person",
        f"LEFT JOIN `{cdr_path}.concept` p_gender_concept ON person.gender_concept_id = p_gender_concept.concept_id",
        f"LEFT JOIN `{cdr_path}.concept` p_race_concept ON person.race_concept_id = p_race_concept.concept_id",
        f"LEFT JOIN `{cdr_path}.concept` p_ethnicity_concept ON person.ethnicity_concept_id = p_ethnicity_concept.concept_id",
        f"LEFT JOIN `{cdr_path}.concept` p_sex_at_birth_concept ON person.sex_at_birth_concept_id = p_sex_at_birth_concept.concept_id",
        f"LEFT JOIN `{cdr_path}.cb_search_person` cb_search_person ON person.person_id = cb_search_person.person_id",
        f"LEFT JOIN `{cdr_path}.observation` p_observation ON person.person_id = p_observation.person_id AND p_observation.observation_concept_id = 1586099"
    ]

    where_conditions = []
    cohort_def = config.get('cohort_definition', {})

    cohort_table_id = cohort_def.get('cohort_table_id')
    if cohort_table_id:
        where_conditions.append(f"person.person_id IN (SELECT person_id FROM `{cdr_path}.{cohort_table_id}`)")
    else:
        # Placeholder for 'include_concepts' and 'exclude_concepts' from cohort_definition in config
        # For simplicity, base cohort will just be persons with EHR data for now.
        # If you add more cohort_definition logic to config.yaml, implement it here.
        where_conditions.append("cb_search_person.has_ehr_data = 1")
    
    # Add a minimum age filter, e.g., age >= 18
    # where_conditions.append("FLOOR(DATE_DIFF(DATE(CURRENT_DATE),DATE(person.birth_datetime), DAY)/365.25) >= 18")

    final_where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""

    sql_query = f"""
    SELECT
        {', '.join(select_clauses)}
    {os.linesep.join(from_join_clauses)}
    {final_where_clause}
    """
    return sql_query


# --- build_observation_duration_query (REMOVED - now derived in Python post time_0) ---
# The observation duration relative to time_0 is best calculated in Python after time_0 is set.

# --- build_condition_datetime_query (REMOVED - now derived in Python post time_0) ---
# Min/Max condition datetimes for COPD are used in get_all_outcome_events_query,
# and other condition datetimes are handled by individual feature extraction.


# Remaining helper functions (filter_columns, stratify_by_risk) are unchanged
def filter_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Filter dataframe to retain only specified columns."""
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in DataFrame: {missing_columns}. Available columns: {list(df.columns)}")
    return df[columns].copy()

def stratify_by_risk(df: pd.DataFrame, risk_column: str, threshold: float) -> pd.DataFrame:
    """Stratify dataset into high vs. low risk groups based on threshold."""
    if risk_column not in df.columns:
        raise KeyError(f"Risk column '{risk_column}' does not exist in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[risk_column]):
        try:
            df[risk_column] = pd.to_numeric(df[risk_column], errors='coerce')
            if df[risk_column].isnull().all():
                raise ValueError(f"Risk column '{risk_column}' became all NaNs after conversion. It must contain numeric data.")
            logging.warning(f"Risk column '{risk_column}' was converted to numeric dtype.")
        except Exception:
            raise ValueError(f"Risk column '{risk_column}' must contain numeric data and could not be converted.")

    df = df.copy()
    df['risk_group'] = np.where(df[risk_column] >= threshold, 'high', 'low')
    df['risk_group'] = np.where(pd.isna(df[risk_column]), 'unknown', df['risk_group'])
    return df

if __name__ == "__main__":
    # This block is for direct testing of the dataloader.py script
    # It assumes environment variables are set and a config.yaml exists at root.
    # For local testing, ensure WORKSPACE_CDR is mocked/set.

    # IMPORTANT: Adjust this path to your configuration.yaml if running directly
    config_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configuration.yaml')

    # Mock AoU environment variables for local test (if not in Workbench)
    # Be sure to set this to a valid value for your BigQuery project if testing
    # os.environ["WORKSPACE_CDR"] = "all-of-us-research-workbench-####.r2023q3_unzipped_data"

    if "WORKSPACE_CDR" not in os.environ:
        logging.error("ERROR: WORKSPACE_CDR environment variable MUST be set for dataloader.py to run directly.")
        logging.error("Please uncomment and set it in the script or ensure your environment is configured (e.g., in AoU Workbench).")
        exit() # Exit if environment not set for direct run

    try:
        logging.info("Starting direct dataloader.py execution...")
        config = load_configuration(config_file_path)
        logging.info("Configuration loaded successfully.")
        
        logging.info("Loading data from BigQuery with cohort construction and time_0 logic...")
        data_df = load_data_from_bigquery(config)
        
        logging.info(f"Data loaded from BigQuery. Shape: {data_df.shape}")
        if not data_df.empty:
            logging.info("First 5 rows of data:")
            logging.info(data_df.head().to_string()) # Use to_string for better logging
            logging.info("\nColumn names after loading:")
            logging.info(data_df.columns.tolist())
            logging.info(f"\nExample time_0: {data_df['time_0'].iloc[0]}")
            logging.info(f"Total events observed: {data_df['event_observed'].sum()}")
            logging.info(f"Median time to event/censoring: {data_df['time_to_event_days'].median()} days")
        else:
            logging.info("DataFrame is empty.")

    except RuntimeError as e:
        logging.error(f"A data loading or configuration error occurred: {e}")
    except KeyError as e:
        logging.error(f"A column or key error occurred: {e}. This might mean a concept was not found or mapped incorrectly.")
    except EnvironmentError as e:
        logging.error(f"Environment setup error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) # Log full traceback
