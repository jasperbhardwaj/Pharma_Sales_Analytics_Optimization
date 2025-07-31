-- Create the database
CREATE DATABASE IF NOT EXISTS proc_dna_pharma_analytics;
USE proc_dna_pharma_analytics;

-- Drop tables if they exist to allow for clean re-runs
DROP TABLE IF EXISTS HCP_Interactions;
DROP TABLE IF EXISTS Sales_Data;
DROP TABLE IF EXISTS Patient_Journey_Milestones;
DROP TABLE IF EXISTS Patient_Demographics;
DROP TABLE IF EXISTS HCPs;
DROP TABLE IF EXISTS Sales_Reps;
DROP TABLE IF EXISTS Territories;
DROP TABLE IF EXISTS Products;

-- Create Products Table
CREATE TABLE Products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100) NOT NULL,
    therapeutic_area VARCHAR(100),
    launch_date DATE
);

-- Create Territories Table
CREATE TABLE Territories (
    territory_id INT PRIMARY KEY AUTO_INCREMENT,
    territory_name VARCHAR(100) NOT NULL,
    geographic_area VARCHAR(100), -- e.g., "North", "South", "East", "West"
    total_population INT,
    target_patient_population INT,
    potential_revenue DECIMAL(15, 2) -- Simulated potential revenue for the territory
);

-- Create Sales_Reps Table
CREATE TABLE Sales_Reps (
    sales_rep_id INT PRIMARY KEY AUTO_INCREMENT,
    rep_name VARCHAR(100) NOT NULL,
    region_assigned VARCHAR(100)
);

-- Create HCPs (Healthcare Professionals) Table
CREATE TABLE HCPs (
    hcp_id INT PRIMARY KEY AUTO_INCREMENT,
    hcp_name VARCHAR(100) NOT NULL,
    specialty VARCHAR(100), -- e.g., "Cardiologist", "Oncologist", "General Physician"
    clinic_location_city VARCHAR(100),
    clinic_location_state VARCHAR(100),
    primary_territory_id INT,
    FOREIGN KEY (primary_territory_id) REFERENCES Territories(territory_id)
);

-- Create Sales_Data Table
CREATE TABLE Sales_Data (
    sale_id INT PRIMARY KEY AUTO_INCREMENT,
    product_id INT,
    sales_rep_id INT,
    territory_id INT,
    hcp_id INT,
    sale_amount DECIMAL(15, 2) NOT NULL,
    sale_date DATE NOT NULL,
    FOREIGN KEY (product_id) REFERENCES Products(product_id),
    FOREIGN KEY (sales_rep_id) REFERENCES Sales_Reps(sales_rep_id),
    FOREIGN KEY (territory_id) REFERENCES Territories(territory_id),
    FOREIGN KEY (hcp_id) REFERENCES HCPs(hcp_id)
);

-- Create HCP_Interactions Table
CREATE TABLE HCP_Interactions (
    interaction_id INT PRIMARY KEY AUTO_INCREMENT,
    hcp_id INT,
    sales_rep_id INT,
    interaction_date DATE,
    interaction_type VARCHAR(50), -- e.g., "Visit", "Call", "Webinar", "Email"
    duration_minutes INT,
    FOREIGN KEY (hcp_id) REFERENCES HCPs(hcp_id),
    FOREIGN KEY (sales_rep_id) REFERENCES Sales_Reps(sales_rep_id)
);

-- Create Patient_Demographics Table (Aggregated/Segmented)
-- This table stores aggregated patient data per territory/disease segment.
CREATE TABLE Patient_Demographics (
    patient_segment_id INT PRIMARY KEY AUTO_INCREMENT,
    territory_id INT,
    disease_area VARCHAR(100), -- e.g., "Diabetes", "Hypertension", "Oncology"
    disease_stage VARCHAR(50), -- e.g., "Early", "Moderate", "Advanced"
    patient_count INT,
    average_treatment_cost DECIMAL(15, 2),
    FOREIGN KEY (territory_id) REFERENCES Territories(territory_id)
);

-- Create Patient_Journey_Milestones Table (Aggregated/Segmented)
-- This table captures average times for patient journey milestones within a segment/territory.
CREATE TABLE Patient_Journey_Milestones (
    milestone_id INT PRIMARY KEY AUTO_INCREMENT,
    patient_segment_id INT,
    milestone_type VARCHAR(100), -- e.g., "Diagnosis to First Rx", "Rx to Adherence", "Relapse"
    average_days_to_milestone INT,
    impact_on_sales_potential DECIMAL(5, 2), -- e.g., a factor affecting sales potential
    FOREIGN KEY (patient_segment_id) REFERENCES Patient_Demographics(patient_segment_id)
);