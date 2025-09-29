-- SQL script to preprocess birth data from raw CSV files

-- -- Define tables for raw data ingestion
-- CREATE TABLE births_old_raw (
--     year INT,
--     month INT,
--     county TEXT,
--     geography_type TEXT,
--     strata TEXT,
--     strata_name TEXT,
--     count TEXT,
--     annotation_code TEXT,
--     annotation_desc TEXT,
--     data_revision_date TEXT
-- );


-- CREATE TABLE births_new_raw (
--     year INT,
--     month INT,
--     county TEXT,
--     geography_type TEXT,
--     strata TEXT,
--     strata_name TEXT,
--     count TEXT,
--     annotation_code TEXT,
--     annotation_desc TEXT,
--     data_extract_date TEXT,
--     data_revision_date TEXT
-- );

-- Import data from CSV files into the staging tables using pgAdmin (Databases -> calhealhdb -> Schemas -> public -> Tables -> births_old_raw or births_new_raw -> Import/Export)

-- -- Clean and transform data
-- CREATE TABLE births_clean AS
-- SELECT
--     year,
--     month,
--     county,
--     COALESCE(NULLIF(count, ''), '0')::INT AS births
-- FROM births_old_raw
-- WHERE strata_name = 'Total Population'

-- UNION ALL

-- SELECT
--     year,
--     month,
--     county,
--     COALESCE(NULLIF(count, ''), '0')::INT AS births
-- FROM births_new_raw
-- WHERE strata_name = 'Total Population';

-- -- Aggregate data to state level (California)
CREATE TABLE births_final AS
SELECT year, month, county, births
FROM births_clean

UNION ALL

SELECT 
    year, 
    month, 
    'California' AS county, 
    SUM(births) AS births
FROM births_clean
GROUP BY year, month;
ORDER BY year, month, county;

-- -- Clean up intermediate tables
-- DROP TABLE births_old_raw;
-- DROP TABLE births_new_raw;
-- DROP TABLE births_clean;
-- DROP TABLE births_state;

-- Save and export the final cleaned data to a CSV file using pgAdmin (Databases -> calhealthdb -> Schemas -> public -> Tables -> births_final -> Import/Export)