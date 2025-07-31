import pandas as pd
import numpy as np
from faker import Faker
import mysql.connector
from datetime import datetime, timedelta
import random

# --- Configuration ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root', # <--- IMPORTANT: Change this
    'password': '12215136@Sql', # <--- IMPORTANT: Change this
    'database': 'proc_dna_pharma_analytics'
}

NUM_PRODUCTS = 10
NUM_TERRITORIES = 15
NUM_SALES_REPS = 30
NUM_HCPS = 500
NUM_SALES_RECORDS = 10000 # Number of sales transactions
NUM_HCP_INTERACTIONS = 15000
NUM_PATIENT_SEGMENTS = 50 # Patient segments per territory * disease area * stage

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

Faker.seed(42) # for reproducibility
fake = Faker('en_IN') # Using Indian locale for more relevant names/cities

# --- 1. Data Generation Functions ---

def generate_products(num):
    products = []
    therapeutic_areas = ['Oncology', 'Cardiology', 'Diabetes', 'Neurology', 'Infectious Diseases', 'Dermatology']
    for i in range(num):
        product_name = f"{fake.word().capitalize()} {fake.word().capitalize()} Drug"
        products.append({
            'product_id': i + 1,
            'product_name': product_name,
            'therapeutic_area': random.choice(therapeutic_areas),
            'launch_date': fake.date_between(start_date='-5y', end_date='today')
        })
    return pd.DataFrame(products)

def generate_territories(num):
    territories = []
    regions = ['North', 'South', 'East', 'West', 'Central']
    cities = [fake.city() for _ in range(num)] # Generate unique-ish cities
    for i in range(num):
        territories.append({
            'territory_id': i + 1,
            'territory_name': f"Territory {i+1} - {cities[i]}",
            'geographic_area': random.choice(regions),
            'total_population': random.randint(100000, 5000000),
            'target_patient_population': random.randint(5000, 100000),
            'potential_revenue': round(random.uniform(500000, 50000000), 2)
        })
    return pd.DataFrame(territories)

def generate_sales_reps(num):
    reps = []
    regions = ['North', 'South', 'East', 'West', 'Central'] # Match territories
    for i in range(num):
        reps.append({
            'sales_rep_id': i + 1,
            'rep_name': fake.name(),
            'region_assigned': random.choice(regions)
        })
    return pd.DataFrame(reps)

def generate_hcps(num, territories_df):
    hcps = []
    specialties = ['Cardiologist', 'Oncologist', 'Neurologist', 'General Physician', 'Dermatologist', 'Endocrinologist']
    territory_ids = territories_df['territory_id'].tolist()
    for i in range(num):
        city = fake.city()
        state = fake.state()
        hcps.append({
            'hcp_id': i + 1,
            'hcp_name': fake.name(),
            'specialty': random.choice(specialties),
            'clinic_location_city': city,
            'clinic_location_state': state,
            'primary_territory_id': random.choice(territory_ids)
        })
    return pd.DataFrame(hcps)

def generate_sales_data(num, products_df, sales_reps_df, territories_df, hcps_df, start_date, end_date):
    sales_data = []
    product_ids = products_df['product_id'].tolist()
    sales_rep_ids = sales_reps_df['sales_rep_id'].tolist()
    territory_ids = territories_df['territory_id'].tolist()
    hcp_ids = hcps_df['hcp_id'].tolist()

    time_delta = (end_date - start_date).days

    for i in range(num):
        sale_date = start_date + timedelta(days=random.randint(0, time_delta))
        sales_data.append({
            'sale_id': i + 1,
            'product_id': random.choice(product_ids),
            'sales_rep_id': random.choice(sales_rep_ids),
            'territory_id': random.choice(territory_ids),
            'hcp_id': random.choice(hcp_ids),
            'sale_amount': round(random.uniform(100, 5000), 2),
            'sale_date': sale_date
        })
    return pd.DataFrame(sales_data)

def generate_hcp_interactions(num, hcps_df, sales_reps_df, start_date, end_date):
    interactions = []
    hcp_ids = hcps_df['hcp_id'].tolist()
    sales_rep_ids = sales_reps_df['sales_rep_id'].tolist()
    interaction_types = ['Visit', 'Call', 'Webinar', 'Email']

    time_delta = (end_date - start_date).days

    for i in range(num):
        interaction_date = start_date + timedelta(days=random.randint(0, time_delta))
        interactions.append({
            'interaction_id': i + 1,
            'hcp_id': random.choice(hcp_ids),
            'sales_rep_id': random.choice(sales_rep_ids),
            'interaction_date': interaction_date,
            'interaction_type': random.choice(interaction_types),
            'duration_minutes': random.randint(5, 60) if random.random() > 0.1 else None # 10% null duration
        })
    return pd.DataFrame(interactions)

def generate_patient_demographics(num_segments, territories_df):
    patient_segments = []
    territory_ids = territories_df['territory_id'].tolist()
    disease_areas = ['Diabetes', 'Hypertension', 'Oncology', 'Asthma', 'Arthritis']
    disease_stages = ['Early', 'Moderate', 'Advanced']

    for i in range(num_segments):
        patient_segments.append({
            'patient_segment_id': i + 1,
            'territory_id': random.choice(territory_ids),
            'disease_area': random.choice(disease_areas),
            'disease_stage': random.choice(disease_stages),
            'patient_count': random.randint(100, 5000),
            'average_treatment_cost': round(random.uniform(500, 10000), 2)
        })
    return pd.DataFrame(patient_segments)

def generate_patient_journey_milestones(patient_demographics_df):
    milestones = []
    milestone_types = {
        'Diabetes': ["Diagnosis to First Rx", "First Rx to Adherence", "Adherence to HbA1c Control"],
        'Oncology': ["Diagnosis to Treatment Initiation", "First Line to Second Line Therapy", "Remission/Relapse"],
        'Hypertension': ["Diagnosis to BP Control", "Annual Follow-up", "Medication Change"],
        'Asthma': ["Diagnosis to Inhaler Use", "Exacerbation Management", "Annual Review"],
        'Arthritis': ["Diagnosis to Pain Management", "Physio Referral", "Surgery Decision"]
    }

    segment_ids = patient_demographics_df['patient_segment_id'].tolist()

    for _, row in patient_demographics_df.iterrows():
        segment_id = row['patient_segment_id']
        disease_area = row['disease_area']

        # Get relevant milestone types for the disease area
        current_milestone_types = milestone_types.get(disease_area, ["General Milestone 1", "General Milestone 2"])

        for mt_type in current_milestone_types:
            milestones.append({
                'patient_segment_id': segment_id,
                'milestone_type': mt_type,
                'average_days_to_milestone': random.randint(30, 730), # 1 month to 2 years
                'impact_on_sales_potential': round(random.uniform(0.5, 1.5), 2) # Factor affecting sales potential
            })
    return pd.DataFrame(milestones)

# --- 2. Database Connection and Insertion ---

def connect_db():
    """Establishes a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            print("Successfully connected to MySQL database.")
            return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        return None

# THIS IS THE CORRECTED INSERT_DATA FUNCTION TO COPY AND PASTE
def insert_data(conn, df, table_name):
    """Inserts DataFrame records into the specified MySQL table."""
    if df.empty:
        print(f"DataFrame for {table_name} is empty. Skipping insertion.")
        return

    cursor = conn.cursor()
    columns = ', '.join(df.columns) # Correct: This uses the actual DataFrame column names
    placeholders = ', '.join(['%s'] * len(df.columns))
    sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    # Prepare data for executemany, handling potential NaNs (empty values)
    # We explicitly convert any pandas NaN/NaT/NA to Python's None,
    # which mysql.connector correctly interprets as SQL NULL.
    data_to_insert = []
    for _, row in df.iterrows():
        processed_row = []
        for item in row:
            if pd.isna(item): # Check for any pandas 'empty' representations
                processed_row.append(None)
            elif isinstance(item, pd.Timestamp): # Handle pandas datetime objects
                processed_row.append(item.strftime('%Y-%m-%d')) # Format for DATE column
            elif isinstance(item, datetime): # Handle standard datetime objects
                processed_row.append(item.strftime('%Y-%m-%d %H:%M:%S')) # Format for DATETIME column
            else:
                processed_row.append(item)
        data_to_insert.append(processed_row)

    try:
        cursor.executemany(sql, data_to_insert)
        conn.commit()
        print(f"Successfully inserted {len(df)} records into {table_name}.")
    except mysql.connector.Error as err:
        print(f"Error inserting data into {table_name}: {err}")
        conn.rollback() # Rollback on error to keep database consistent
    finally:
        cursor.close()

# --- Main Simulation and Insertion Logic ---
if __name__ == '__main__':
    print("--- Generating Simulated Data ---")
    products_df = generate_products(NUM_PRODUCTS)
    territories_df = generate_territories(NUM_TERRITORIES)
    sales_reps_df = generate_sales_reps(NUM_SALES_REPS)
    hcps_df = generate_hcps(NUM_HCPS, territories_df)
    sales_data_df = generate_sales_data(NUM_SALES_RECORDS, products_df, sales_reps_df, territories_df, hcps_df, START_DATE, END_DATE)
    hcp_interactions_df = generate_hcp_interactions(NUM_HCP_INTERACTIONS, hcps_df, sales_reps_df, START_DATE, END_DATE)
    patient_demographics_df = generate_patient_demographics(NUM_PATIENT_SEGMENTS, territories_df)
    patient_journey_milestones_df = generate_patient_journey_milestones(patient_demographics_df)

    print("\n--- Connecting to Database and Inserting Data ---")
    conn = connect_db()
    if conn:
        insert_data(conn, products_df, 'Products')
        insert_data(conn, territories_df, 'Territories')
        insert_data(conn, sales_reps_df, 'Sales_Reps')
        insert_data(conn, hcps_df, 'HCPs')
        insert_data(conn, sales_data_df, 'Sales_Data')
        insert_data(conn, hcp_interactions_df, 'HCP_Interactions')
        insert_data(conn, patient_demographics_df, 'Patient_Demographics')
        insert_data(conn, patient_journey_milestones_df, 'Patient_Journey_Milestones')
        conn.close()
        print("Database connection closed.")
    else:
        print("Skipping data insertion due to database connection error.")

    print("\n--- Data Simulation and Insertion Complete ---")