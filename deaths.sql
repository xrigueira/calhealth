-- SQL script to preprocess death data from raw CSV files

-- -- Define final table structure
-- CREATE TABLE deaths_raw (
--     year INT,
--     month INT,
--     county TEXT,
--     strata TEXT,
--     strata_name TEXT,
--     cause TEXT,
--     deaths INT
-- );

-- -- Create staging tables for raw data
-- CREATE TABLE deaths_staging_1 (
--     year INT,
--     month INT,
--     county TEXT,
--     geography_type TEXT,
--     strata TEXT,
--     strata_name TEXT,
--     cause TEXT,
--     cause_desc TEXT,
--     icd_revision TEXT,
--     count TEXT, -- note: comes as text because of empty/suppressed values
--     annotation_code TEXT,
--     annotation_desc TEXT,
--     data_revision_date TEXT
-- );

-- CREATE TABLE deaths_staging_2 (
--     year INT,
--     month INT,
--     county TEXT,
--     geography_type TEXT,
--     strata TEXT,
--     strata_name TEXT,
--     cause TEXT,
--     cause_desc TEXT,
--     icd_revision TEXT,
--     count TEXT, -- note: comes as text because of empty/suppressed values
--     annotation_code TEXT,
--     annotation_desc TEXT,
--     data_revision_date TEXT
-- );

-- CREATE TABLE deaths_staging_3 (
--     year INT,
--     month INT,
--     county TEXT,
--     geography_type TEXT,
--     strata TEXT,
--     strata_name TEXT,
--     cause TEXT,
--     cause_desc TEXT,
--     icd_revision TEXT,
--     count TEXT, -- note: comes as text because of empty/suppressed values
--     annotation_code TEXT,
--     annotation_desc TEXT,
--     data_extract_date TEXT,
--     data_revision_date TEXT
-- );

-- Import data from CSV files into the staging tables using pgAdmin (Databases -> calhealhdb -> Schemas -> public -> Tables -> births_old_raw or births_new_raw -> Import/Export)

-- Clean and transform data
INSERT INTO deaths_raw (year, month, county, strata, strata_name, cause, deaths)
SELECT
    year,
    month,
    county,
    strata,
    strata_name,
    cause_desc AS cause,
    COALESCE(NULLIF(count, ''), '0')::INT AS deaths
FROM deaths_staging_1
WHERE year IS NOT NULL; -- Avoid header rows

INSERT INTO deaths_raw (year, month, county, strata, strata_name, cause, deaths)
SELECT
    year,
    month,
    county,
    strata,
    strata_name,
    cause_desc AS cause,
    COALESCE(NULLIF(count, ''), '0')::INT AS deaths
FROM deaths_staging_2
WHERE year IS NOT NULL; -- Avoid header rows

INSERT INTO deaths_raw (year, month, county, strata, strata_name, cause, deaths)
SELECT
    year,
    month,
    county,
    strata,
    strata_name,
    cause_desc AS cause,
    COALESCE(NULLIF(count, ''), '0')::INT AS deaths
FROM deaths_staging_3
WHERE year IS NOT NULL; -- Avoid header rows

-- Clean up intermediate tables
-- DROP TABLE deaths_staging_1;
-- DROP TABLE deaths_staging_2;
-- DROP TABLE deaths_staging_3;

-- Save and export the final cleaned data to a CSV file using pgAdmin (Databases -> calhealthdb -> Schemas -> public -> Tables -> births_final -> Import/Export)