# CalHealth

Modeling health data from the US and California with data-driven algorithms.

1. Downloaded raw data from these sources:
    - [Microsoft length of stay](https://www.kaggle.com/datasets/aayushchou/hospital-length-of-stay-dataset-microsoft)
    - [CDPH births](https://data.ca.gov/dataset/live-birth-profiles-by-county)
    - [CDPH deaths](https://data.ca.gov/dataset/death-profiles-by-county)

2. Preprocess
    - Length of stay: convert rcount, gender, and facid to numeric, and drop date columns.
    - Births: build the births dataset by joining data from 1960 to 2023 (first file) with data from 2025 to 2025 (second file) on SQL.
    This involves creating the stating tables, importing the CSV data into such tables, removing unnecessary columns (Strata_Name, Annotation_Code, Annotation_Desc, and Data_Revision_Date), performing a union, imputing null (cell-supression) values with 0, and getting state totals.
    - Deaths: build the deaths dataset by joining data from 2014 to 2018 (first file), 2019 to 2023 (second file), and 2025 to 2025 (third file) on SQL.
    This involved creating the stating tables, importing the CSV data into such tables, removing unnecessary columns (Geography_Type, ICD_Revision, Annotation_Code, Annotation_Desc, Data_Revision_Date), performing a union, and imputing null (cell-supression) values with 0.

3. Modeling
    - Length of stay: Build the traing and testing sets with a 80/20 split and apply standard scaling. Build Random Forest model (100 regression trees). Train, test, and evaluate resutls.
    - Births: Build training and testing sets. Build neural network.

4. Dashboard
    4.1 Created a California map displaying the number of deaths by year, month, strata, and cause.
    4.2 Built a stacked bar chart showing demographic and cause breakdowns as a function of year, month, and county.
    4.3 Combined both visualizations into a dashboard, applying year, month, starta, and cause filters to both the map and bar chart.
    4.4 Finally, added a filter action that allows users to click on a county (in the map) and automatically update the demographic breakdown (bar chart) based on the selection.
    4.5 [Published](https://public.tableau.com/app/profile/xurxo.rigueira/viz/deaths_dash/Dashboard) Tableau Public.