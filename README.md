# -*- coding: utf-8 -*-
"""
COVID-19 Global Data Tracker Project
"""

# ## Project Description:
# In this project, we will build a data analysis and reporting notebook that tracks global COVID-19 trends.
# The project will analyze cases, deaths, recoveries, and vaccinations across countries and time.
# We will clean and process real-world data, perform exploratory data analysis (EDA), generate insights,
# and visualize trends using Python data tools.
#
# By the end, we’ll have a data analysis report with visuals and narrative insights, suitable for presentation.

# ## 🚩 Project Objectives:
# ✅ Import and clean COVID-19 global data
# ✅ Analyze time trends (cases, deaths, vaccinations)
# ✅ Compare metrics across countries/regions (with a focus on Kenya, USA, India)
# ✅ Visualize trends with charts and maps
# ✅ Communicate findings in a Jupyter Notebook

# ## 🗂️ Project Segments (Step-by-Step Guide)

# ### 1️⃣ Data Collection
# **Goal:** Obtain a reliable COVID-19 dataset.
#
# **✅ Data Sources:** Our World in Data COVID-19 Dataset (CSV)
#
# **✅ Action:** Download `owid-covid-data.csv` from [https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-data.csv](https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-data.csv) and save it in your working directory.

# ### 2️⃣ Data Loading & Exploration

import pandas as pd

# Load the dataset
try:
    df = pd.read_csv('owid-covid-data.csv')
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: owid-covid-data.csv not found in the working directory. Please download it.")
    exit()

# Check columns
print("\nColumns in the dataset:")
print(df.columns)
print("\n")

# Preview the first few rows
print("First 5 rows of the dataset:")
print(df.head())
print("\n")

# Identify missing values
print("Missing values per column:")
print(df.isnull().sum())
print("\n")

# Get data types of each column
print("Data types of each column:")
print(df.dtypes)

# ### 3️⃣ Data Cleaning

# Filter for countries of interest (Kenya, USA, India)
countries_of_interest = ['Kenya', 'United States', 'India']
df_filtered = df[df['location'].isin(countries_of_interest)].copy()

# Convert date column to datetime
df_filtered['date'] = pd.to_datetime(df_filtered['date'])

# Set 'date' as index for time-series analysis
df_filtered.set_index('date', inplace=True)

# Drop rows with missing dates (shouldn't be any after loading correctly)
df_filtered.dropna(subset=['location'], inplace=True)

# Handle missing numeric values (forward fill, then fill remaining with 0)
numeric_cols = df_filtered.select_dtypes(include=['number']).columns
df_filtered[numeric_cols] = df_filtered[numeric_cols].fillna(method='ffill')
df_filtered.fillna(0, inplace=True)

# Reset index
df_filtered.reset_index(inplace=True)

print("\nCleaned and filtered data:")
print(df_filtered.head())
print("\nMissing values after cleaning:")
print(df_filtered.isnull().sum())

# ### 4️⃣ Exploratory Data Analysis (EDA)

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Plot total cases over time for selected countries
plt.figure(figsize=(12, 6))
for country in countries_of_interest:
    country_data = df_filtered[df_filtered['location'] == country]
    plt.plot(country_data['date'], country_data['total_cases'], label=country)
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.title("Total COVID-19 Cases Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot total deaths over time
plt.figure(figsize=(12, 6))
for country in countries_of_interest:
    country_data = df_filtered[df_filtered['location'] == country]
    plt.plot(country_data['date'], country_data['total_deaths'], label=country)
plt.xlabel("Date")
plt.ylabel("Total Deaths")
plt.title("Total COVID-19 Deaths Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compare daily new cases between countries
plt.figure(figsize=(12, 6))
for country in countries_of_interest:
    country_data = df_filtered[df_filtered['location'] == country]
    plt.plot(country_data['date'], country_data['new_cases'], label=country)
plt.xlabel("Date")
plt.ylabel("Daily New Cases")
plt.title("Daily New COVID-19 Cases")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate the death rate (total_deaths / total_cases)
df_filtered['death_rate'] = df_filtered['total_deaths'] / df_filtered['total_cases']

# Plot death rate over time
plt.figure(figsize=(12, 6))
for country in countries_of_interest:
    country_data = df_filtered[df_filtered['location'] == country]
    plt.plot(country_data['date'], country_data['death_rate'], label=country)
plt.xlabel("Date")
plt.ylabel("Death Rate (Total Deaths / Total Cases)")
plt.title("COVID-19 Death Rate Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar chart of top countries by total cases (latest date)
latest_data = df[df['date'] == df['date'].max()]
top_n = 10
top_cases = latest_data.nlargest(top_n, 'total_cases')

plt.figure(figsize=(10, 6))
sns.barplot(x='total_cases', y='location', data=top_cases)
plt.xlabel("Total Cases")
plt.ylabel("Country")
plt.title(f"Top {top_n} Countries by Total COVID-19 Cases (as of {df['date'].max().strftime('%Y-%m-%d')})")
plt.tight_layout()
plt.show()

# ### 5️⃣ Visualizing Vaccination Progress

# Plot cumulative vaccinations over time for selected countries
plt.figure(figsize=(12, 6))
for country in countries_of_interest:
    country_data = df_filtered[df_filtered['location'] == country].dropna(subset=['total_vaccinations'])
    plt.plot(country_data['date'], country_data['total_vaccinations'], label=country)
plt.xlabel("Date")
plt.ylabel("Total Vaccinations")
plt.title("Cumulative COVID-19 Vaccinations Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compare % vaccinated population (using 'people_vaccinated_per_hundred')
plt.figure(figsize=(12, 6))
for country in countries_of_interest:
    country_data = df_filtered[df_filtered['location'] == country].dropna(subset=['people_vaccinated_per_hundred'])
    plt.plot(country_data['date'], country_data['people_vaccinated_per_hundred'], label=country)
plt.xlabel("Date")
plt.ylabel("Percentage of Population Vaccinated")
plt.title("Percentage of Population Vaccinated Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Optional: Pie chart for the latest vaccination percentages
latest_vaccination = df[df['date'] == df['date'].max()].dropna(subset=['people_vaccinated_per_hundred'])
vaccination_subset = latest_vaccination[latest_vaccination['location'].isin(countries_of_interest)]

if not vaccination_subset.empty:
    plt.figure(figsize=(8, 8))
    plt.pie(vaccination_subset['people_vaccinated_per_hundred'], labels=vaccination_subset['location'], autopct='%1.1f%%', startangle=140)
    plt.title(f"Percentage of Population Vaccinated (as of {df['date'].max().strftime('%Y-%m-%d')})")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
else:
    print("No vaccination data available for the selected countries on the latest date.")

# ### 6️⃣ Optional: Build a Choropleth Map

import plotly.express as px

# Prepare a dataframe with iso_code, total_cases for the latest date
latest_data_world = df[df['date'] == df['date'].max()].dropna(subset=['iso_code', 'total_cases'])

# Plot a choropleth showing total cases
fig_cases = px.choropleth(latest_data_world,
                     locations="iso_code",
                     color="total_cases",
                     hover_name="location",
                     color_continuous_scale=px.colors.sequential.Plasma,
                     title=f"Total COVID-19 Cases by Country (as of {df['date'].max().strftime('%Y-%m-%d')})")
fig_cases.show()

# Prepare a dataframe with iso_code, people_vaccinated_per_hundred for the latest date
latest_vaccination_world = df[df['date'] == df['date'].max()].dropna(subset=['iso_code', 'people_vaccinated_per_hundred'])

# Plot a choropleth showing vaccination rates
fig_vaccination = px.choropleth(latest_vaccination_world,
                     locations="iso_code",
                     color="people_vaccinated_per_hundred",
                     hover_name="location",
                     color_continuous_scale=px.colors.sequential.Greens,
                     title=f"Percentage of Population Vaccinated by Country (as of {df['date'].max().strftime('%Y-%m-%d')})")
fig_vaccination.show()

# ### 7️⃣ Insights & Reporting

# **Key Insights:**
#
# 1.  **Total Cases:** The United States has consistently shown a significantly higher number of total COVID-19 cases compared to Kenya and India throughout the pandemic.
# 2.  **Death Trends:** The trend of total deaths also follows a similar pattern, with the United States recording the highest numbers among the selected countries. However, the death rates (proportion of total cases resulting in death) might show a different picture.
# 3.  **Daily New Cases:** Examining daily new cases reveals the different waves and peaks of the pandemic in each country at different times. Kenya's waves might have been less pronounced or occurred at different periods compared to the USA and India.
# 4.  **Vaccination Progress:** The vaccination rollout speed and coverage vary significantly. Comparing the cumulative vaccinations and the percentage of the population vaccinated provides insights into the progress made by each country. Kenya's vaccination efforts can be compared to the other two nations.
# 5.  **Global Disparities:** The choropleth maps highlight the global distribution of total cases and vaccination rates, often revealing disparities between different regions and countries.

# **Anomalies and Interesting Patterns:**
#
# * Identify any sudden spikes or drops in the daily new cases for each country, which might correspond to specific events or reporting changes.
# * Analyze periods where the death rate showed significant fluctuations, potentially indicating changes in healthcare capacity or the emergence of new variants.
# * Observe the inflection points in the vaccination curves, indicating when the vaccination campaigns started gaining momentum in each country.

# **Further Exploration (Optional):**
#
# * Analyze other metrics available in the dataset, such as hospitalizations, ICU patients, or tests conducted.
# * Compare the impact of government stringency index on the spread of the virus in the selected countries.
# * Investigate regional variations within each country if the dataset provides that level of detail.

# **Conclusion:**
# This analysis provides a comparative overview of the COVID-19 pandemic's progression in Kenya, the United States, and India, as well as a global perspective on total cases and vaccination rates. The visualizations and insights highlight the varying impacts and responses to the pandemic across different regions. Further investigation into specific factors and additional data could provide a more nuanced understanding of the pandemic's dynamics.
