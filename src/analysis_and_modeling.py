import pandas as pd
import numpy as np
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

# --- Configuration ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root', # <--- Your actual MySQL user
    'password': '12215136@Sql', # <--- Your actual MySQL password
    'database': 'proc_dna_pharma_analytics'
}

# --- Database Connection and Data Fetching ---

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

def fetch_data(conn, table_name):
    """Fetches all data from a specified table."""
    query = f"SELECT * FROM {table_name}"
    try:
        df = pd.read_sql(query, conn)
        print(f"Fetched {len(df)} records from {table_name}.")
        return df
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- Main Analysis and Modeling Logic ---
if __name__ == '__main__':
    # ALL THE CODE BELOW THIS LINE (until the very end of the file)
    # MUST BE INDENTED BY 4 SPACES (ONE TAB LEVEL)
    conn = connect_db()
    if not conn:
        print("Exiting: Could not connect to the database.")
        exit()

    # Fetch all necessary data
    products_df = fetch_data(conn, 'Products')
    territories_df = fetch_data(conn, 'Territories')
    sales_reps_df = fetch_data(conn, 'Sales_Reps')
    hcps_df = fetch_data(conn, 'HCPs')
    sales_data_df = fetch_data(conn, 'Sales_Data')
    sales_data_df['sale_date'] = pd.to_datetime(sales_data_df['sale_date']) # This line was added in previous fix
    hcp_interactions_df = fetch_data(conn, 'HCP_Interactions')
    patient_demographics_df = fetch_data(conn, 'Patient_Demographics')
    patient_journey_milestones_df = fetch_data(conn, 'Patient_Journey_Milestones')

    conn.close()
    print("Database connection closed after fetching data.")

    # --- 1. Data Merging and Feature Engineering ---
    print("\n--- Performing Data Merging and Feature Engineering ---")

    # Merge sales data with relevant dimensions
    df = sales_data_df.merge(products_df, on='product_id', how='left')
    df = df.merge(sales_reps_df, on='sales_rep_id', how='left')
    df = df.merge(territories_df, on='territory_id', how='left')
    df = df.merge(hcps_df, on='hcp_id', how='left')

    # Feature Engineering from Sales Data
    df['sale_month'] = df['sale_date'].dt.month
    df['sale_quarter'] = df['sale_date'].dt.quarter
    df['sale_day_of_week'] = df['sale_date'].dt.dayofweek # Monday=0, Sunday=6
    df['sale_year'] = df['sale_date'].dt.year

    # Aggregate HCP Interaction Features
    hcp_interaction_summary = hcp_interactions_df.groupby('hcp_id').agg(
        total_interactions=('interaction_id', 'count'),
        avg_duration_minutes=('duration_minutes', 'mean')
    ).reset_index()
    df = df.merge(hcp_interaction_summary, on='hcp_id', how='left').fillna(0) # Fill NaN for HCPs with no interactions

    # Aggregate Patient Demographics Features at Territory Level
    territory_patient_agg = patient_demographics_df.groupby('territory_id').agg(
        total_territory_patients=('patient_count', 'sum'),
        avg_territory_treatment_cost=('average_treatment_cost', 'mean')
    ).reset_index()
    df = df.merge(territory_patient_agg, on='territory_id', how='left').fillna(0)

    # Consider patient journey impact (simple aggregation for now)
    # This is a more complex join, so let's keep it simple for a first pass
    # For a deeper dive, you'd link patient_demographics to products via therapeutic area, etc.
    # For now, we'll just use the aggregated patient counts and potential revenue from territories.

    print("Merged DataFrame head:")
    print(df.head())
    print(f"DataFrame shape after merging and feature engineering: {df.shape}")

    # --- 2. Advanced Exploratory Data Analysis (EDA) ---
    print("\n--- Performing Advanced Exploratory Data Analysis (EDA) ---")

    # Sales Trend Over Time
    monthly_sales = df.groupby(df['sale_date'].dt.to_period('M'))['sale_amount'].sum()
    monthly_sales = monthly_sales.to_timestamp() # Convert PeriodIndex to Timestamp for plotting

    plt.figure(figsize=(14, 7))
    sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker='o')
    plt.title('Monthly Sales Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales Amount')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/monthly_sales_trend.png') # Save plot
    plt.show()

    # Sales Distribution by Therapeutic Area
    sales_by_therapy = df.groupby('therapeutic_area')['sale_amount'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sales_by_therapy.index, y=sales_by_therapy.values, palette='viridis')
    plt.title('Total Sales by Therapeutic Area')
    plt.xlabel('Therapeutic Area')
    plt.ylabel('Total Sales Amount')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/sales_by_therapy.png') # Save plot
    plt.show()

    # Correlation between HCP Interactions and Sales
    hcp_engagement_sales = df.groupby('hcp_id').agg(
        total_sales=('sale_amount', 'sum'),
        total_interactions=('total_interactions', 'first')
    ).reset_index().fillna(0)

    hcp_engagement_sales_filtered = hcp_engagement_sales[(hcp_engagement_sales['total_interactions'] > 0) & (hcp_engagement_sales['total_sales'] > 0)]

    if not hcp_engagement_sales_filtered.empty:
        plt.figure(figsize=(10, 7))
        sns.regplot(x='total_interactions', y='total_sales', data=hcp_engagement_sales_filtered, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        plt.title('Correlation: Total HCP Interactions vs. Total Sales')
        plt.xlabel('Total Interactions (Simulated)')
        plt.ylabel('Total Sales Amount')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('visualizations/hcp_interaction_sales_corr.png') # Save plot
        plt.show()
        print(f"Correlation coefficient between Total Interactions and Total Sales: {hcp_engagement_sales_filtered['total_interactions'].corr(hcp_engagement_sales_filtered['total_sales']):.2f}")
    else:
        print("Not enough data to show correlation between HCP interactions and sales.")


    # Sales Rep Performance
    rep_performance = df.groupby('sales_rep_id').agg(
        total_sales=('sale_amount', 'sum'),
        total_products_sold=('product_id', 'nunique'),
        total_hcps_engaged=('hcp_id', 'nunique')
    ).reset_index()
    rep_performance = rep_performance.merge(sales_reps_df[['sales_rep_id', 'rep_name']], on='sales_rep_id')
    print("\nTop 5 Sales Reps by Total Sales:")
    print(rep_performance.sort_values(by='total_sales', ascending=False).head())

    # --- 3. Predictive Modeling (Sales Amount Prediction) ---
    print("\n--- Building Predictive Model for Sales Amount ---")

    # Select features for the model
    features = [
        'sale_month', 'sale_quarter', 'sale_day_of_week', 'sale_year',
        'total_population', 'target_patient_population', 'potential_revenue',
        'total_interactions', 'avg_duration_minutes',
        'total_territory_patients', 'avg_territory_treatment_cost'
    ]
    target = 'sale_amount'

    # Handle categorical features using Label Encoding for simplicity
    categorical_cols = ['therapeutic_area', 'geographic_area', 'specialty']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            features.append(col + '_encoded')

    # Ensure all features exist in the DataFrame
    model_df = df[features + [target]].copy()
    model_df.dropna(inplace=True)

    if model_df.empty:
        print("Not enough clean data to build a predictive model. Skipping model training.")
    else:
        X = model_df[features]
        y = model_df[target]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")

        # Initialize and train a RandomForestRegressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel Evaluation:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R-squared (R2): {r2:.2f}")

        # Feature Importance
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)

        print("\nTop 10 Feature Importances:")
        print(feature_importances.head(10))

        plt.figure(figsize=(12, 7))
        sns.barplot(x='importance', y='feature', data=feature_importances.head(10), palette='magma')
        plt.title('Top 10 Feature Importances in Sales Prediction Model')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('visualizations/feature_importances.png') # Save plot
        plt.show()

        # Actual vs. Predicted Sales Plot
        plt.figure(figsize=(10, 7))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Diagonal line
        plt.title('Actual vs. Predicted Sales Amount')
        plt.xlabel('Actual Sales Amount')
        plt.ylabel('Predicted Sales Amount')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('visualizations/actual_vs_predicted_sales.png') # Save plot
        plt.show()

    print("\n--- Project Analysis and Modeling Complete ---")
    print("Review the generated plots and console output for insights.")