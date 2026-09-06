# Import Libraries

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


# Set Display Options

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 140)

plt.rcParams["figure.figsize"] = (10, 6)


# Load Dataset

file_path = "seasonal_agriculture_performance_dataset (2).csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head().to_string())


# Understand the Dataset

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes.to_string())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe().T.to_string())


# Check Missing Values

print("\nMissing Values Before Cleaning:")

missing_values = df.isnull().sum()

missing_values = missing_values[
    missing_values > 0
]

if len(missing_values) > 0:
    print(missing_values.to_string())
else:
    print("No missing values found.")


# Check Duplicate Records

duplicate_count = df.duplicated().sum()

print("\nNumber of Duplicate Records:")
print(duplicate_count)


# Check Important Categories

print("\nNumber of Unique Values:")

for column in [
    "State",
    "District",
    "Crop",
    "Season",
    "Irrigation_Method"
]:

    if column in df.columns:

        print(
            f"{column}: "
            f"{df[column].nunique()}"
        )


# Data Cleaning

df_clean = df.copy()

print("\nStarting Data Cleaning...")


# Remove Duplicate Records

before_duplicates = len(df_clean)

df_clean = (
    df_clean
    .drop_duplicates()
    .reset_index(drop=True)
)

after_duplicates = len(df_clean)

print(
    "Duplicate records removed:",
    before_duplicates - after_duplicates
)


# Replace Infinite Values

df_clean = df_clean.replace(
    [np.inf, -np.inf],
    np.nan
)


# Identify Numerical Columns

numeric_columns = (
    df_clean
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)


# Handle Missing Numerical Values

for column in numeric_columns:

    if column == "Yield_Tonnes_Ha":
        continue

    if df_clean[column].isnull().sum() > 0:

        median_value = (
            df_clean[column]
            .median()
        )

        df_clean[column] = (
            df_clean[column]
            .fillna(median_value)
        )


# Identify Categorical Columns

categorical_columns = (
    df_clean
    .select_dtypes(include="object")
    .columns
    .tolist()
)


# Handle Missing Categorical Values

for column in categorical_columns:

    if df_clean[column].isnull().sum() > 0:

        mode_value = (
            df_clean[column]
            .mode()[0]
        )

        df_clean[column] = (
            df_clean[column]
            .fillna(mode_value)
        )


# Check Missing Yield Values

print("\nMissing Yield Values:")

print(
    df_clean[
        "Yield_Tonnes_Ha"
    ].isnull().sum()
)


# Check Remaining Missing Values

print("\nMissing Values After Cleaning:")

remaining_missing = (
    df_clean.isnull().sum()
)

remaining_missing = (
    remaining_missing[
        remaining_missing > 0
    ]
)

if len(remaining_missing) > 0:

    print(
        remaining_missing.to_string()
    )

else:

    print(
        "No missing values remain."
    )


# Dataset Overview After Cleaning

print("\nCleaned Dataset Shape:")
print(df_clean.shape)

print("\nCleaned Dataset Preview:")
print(
    df_clean.head().to_string()
)


# Dataset Coverage

print("\nDataset Coverage:")

print(
    "Total Records:",
    len(df_clean)
)

print(
    "States:",
    df_clean["State"].nunique()
)

print(
    "Districts:",
    df_clean["District"].nunique()
)

print(
    "Crops:",
    df_clean["Crop"].nunique()
)

print(
    "Seasons:",
    df_clean["Season"].nunique()
)

print(
    "Irrigation Methods:",
    df_clean["Irrigation_Method"].nunique()
)


# Identify Seasons

print("\nSeasons Present:")

print(
    df_clean["Season"]
    .value_counts()
    .to_string()
)


# Create Season Order

season_order = [
    "Kharif",
    "Rabi",
    "Zaid",
    "Summer",
    "Winter",
    "Monsoon",
    "Autumn",
    "Spring"
]

existing_seasons = [
    season
    for season in season_order
    if season in df_clean["Season"].unique()
]

other_seasons = [
    season
    for season in df_clean["Season"].unique()
    if season not in existing_seasons
]

final_season_order = (
    existing_seasons +
    other_seasons
)


# Seasonal Distribution

season_count = (
    df_clean["Season"]
    .value_counts()
    .reindex(final_season_order)
    .dropna()
)

print("\nNumber of Records by Season:")
print(
    season_count.to_string()
)


# Visualization 1: Bar Chart
# Number of Records by Season

plt.figure(figsize=(10, 6))

bars = plt.bar(
    season_count.index,
    season_count.values
)

plt.title(
    "Number of Agricultural Records by Season"
)

plt.xlabel("Season")
plt.ylabel("Number of Records")

for bar, value in zip(
    bars,
    season_count.values
):

    plt.text(
        bar.get_x() +
        bar.get_width() / 2,
        bar.get_height(),
        f"{int(value)}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# Visualization 2: Pie Chart
# Share of Records by Season

plt.figure(figsize=(8, 8))

plt.pie(
    season_count.values,
    labels=season_count.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Distribution of Agricultural Records Across Seasons"
)

plt.tight_layout()
plt.show()


# Seasonal Performance Analysis

season_summary = (
    df_clean
    .groupby("Season")
    .agg(
        Records=(
            "Farm_ID",
            "count"
        ),
        Average_Yield=(
            "Yield_Tonnes_Ha",
            "mean"
        ),
        Total_Production=(
            "Production_Tonnes",
            "sum"
        ),
        Average_Rainfall=(
            "Rainfall_mm",
            "mean"
        ),
        Average_Temperature=(
            "Avg_Temperature_C",
            "mean"
        ),
        Average_Humidity=(
            "Humidity_pct",
            "mean"
        ),
        Average_Soil_Moisture=(
            "Soil_Moisture_pct",
            "mean"
        ),
        Average_Water_Used=(
            "Water_Used_m3",
            "mean"
        ),
        Average_Revenue=(
            "Revenue_INR",
            "mean"
        ),
        Average_Cost=(
            "Total_Cost_INR",
            "mean"
        ),
        Average_Profit=(
            "Profit_INR",
            "mean"
        )
    )
    .reindex(final_season_order)
)

print("\nSeasonal Performance Summary:")

print(
    season_summary
    .round(2)
    .to_string()
)


# Visualization 3: Bar Chart
# Average Yield by Season

season_yield = (
    df_clean
    .groupby("Season")
    ["Yield_Tonnes_Ha"]
    .mean()
    .reindex(final_season_order)
)

print("\nAverage Yield by Season:")

print(
    season_yield
    .round(2)
    .to_string()
)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    season_yield.index,
    season_yield.values
)

plt.title(
    "Average Yield by Season"
)

plt.xlabel("Season")
plt.ylabel(
    "Average Yield (Tonnes/Ha)"
)

for bar, value in zip(
    bars,
    season_yield.values
):

    plt.text(
        bar.get_x() +
        bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# Total Production by Season

season_production = (
    df_clean
    .groupby("Season")
    ["Production_Tonnes"]
    .sum()
    .reindex(final_season_order)
)

print("\nTotal Production by Season:")

print(
    season_production
    .round(2)
    .to_string()
)


# Visualization 4: Bar Chart
# Total Production by Season

plt.figure(figsize=(10, 6))

bars = plt.bar(
    season_production.index,
    season_production.values
)

plt.title(
    "Total Agricultural Production by Season"
)

plt.xlabel("Season")
plt.ylabel(
    "Total Production (Tonnes)"
)

for bar, value in zip(
    bars,
    season_production.values
):

    plt.text(
        bar.get_x() +
        bar.get_width() / 2,
        bar.get_height(),
        f"{value:,.0f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# Visualization 5: Box Plot
# Yield Variation Across Seasons

season_yield_data = []

season_labels = []

for season in final_season_order:

    values = (
        df_clean[
            df_clean["Season"] == season
        ]["Yield_Tonnes_Ha"]
        .dropna()
    )

    if len(values) > 0:

        season_yield_data.append(
            values
        )

        season_labels.append(
            season
        )

plt.figure(figsize=(10, 6))

plt.boxplot(
    season_yield_data,
    labels=season_labels
)

plt.title(
    "Yield Variation Across Seasons"
)

plt.xlabel("Season")
plt.ylabel(
    "Yield (Tonnes/Ha)"
)

plt.tight_layout()
plt.show()


# Seasonal Economic Analysis

economic_summary = (
    df_clean
    .groupby("Season")
    .agg(
        Average_Revenue=(
            "Revenue_INR",
            "mean"
        ),
        Average_Cost=(
            "Total_Cost_INR",
            "mean"
        ),
        Average_Profit=(
            "Profit_INR",
            "mean"
        )
    )
    .reindex(final_season_order)
)

print("\nEconomic Performance by Season:")

print(
    economic_summary
    .round(2)
    .to_string()
)


# Visualization 6: Grouped Bar Chart
# Revenue, Cost and Profit by Season

x = np.arange(
    len(economic_summary.index)
)

width = 0.25

plt.figure(figsize=(12, 6))

plt.bar(
    x - width,
    economic_summary[
        "Average_Revenue"
    ],
    width,
    label="Revenue"
)

plt.bar(
    x,
    economic_summary[
        "Average_Cost"
    ],
    width,
    label="Cost"
)

plt.bar(
    x + width,
    economic_summary[
        "Average_Profit"
    ],
    width,
    label="Profit"
)

plt.title(
    "Average Revenue, Cost and Profit by Season"
)

plt.xlabel("Season")

plt.ylabel(
    "Amount (INR)"
)

plt.xticks(
    x,
    economic_summary.index
)

plt.legend()

plt.tight_layout()
plt.show()


# Crop Performance Analysis

crop_summary = (
    df_clean
    .groupby("Crop")
    .agg(
        Average_Yield=(
            "Yield_Tonnes_Ha",
            "mean"
        ),
        Average_Profit=(
            "Profit_INR",
            "mean"
        ),
        Average_Revenue=(
            "Revenue_INR",
            "mean"
        ),
        Average_Water_Used=(
            "Water_Used_m3",
            "mean"
        ),
        Records=(
            "Farm_ID",
            "count"
        )
    )
    .sort_values(
        "Average_Yield",
        ascending=False
    )
)

print("\nCrop Performance Summary:")

print(
    crop_summary
    .round(2)
    .to_string()
)


# Visualization 7: Horizontal Bar Chart
# Average Yield by Crop

crop_yield_plot = (
    crop_summary
    .sort_values(
        "Average_Yield"
    )
)

plt.figure(figsize=(12, 7))

bars = plt.barh(
    crop_yield_plot.index,
    crop_yield_plot[
        "Average_Yield"
    ]
)

plt.title(
    "Average Yield by Crop"
)

plt.xlabel(
    "Average Yield (Tonnes/Ha)"
)

plt.ylabel("Crop")

for bar, value in zip(
    bars,
    crop_yield_plot[
        "Average_Yield"
    ]
):

    plt.text(
        value,
        bar.get_y() +
        bar.get_height() / 2,
        f" {value:.2f}",
        va="center"
    )

plt.tight_layout()
plt.show()


# Visualization 8: Horizontal Bar Chart
# Average Profit by Crop

crop_profit_plot = (
    crop_summary
    .sort_values(
        "Average_Profit"
    )
)

plt.figure(figsize=(12, 7))

bars = plt.barh(
    crop_profit_plot.index,
    crop_profit_plot[
        "Average_Profit"
    ]
)

plt.title(
    "Average Profit by Crop"
)

plt.xlabel(
    "Average Profit (INR)"
)

plt.ylabel("Crop")

for bar, value in zip(
    bars,
    crop_profit_plot[
        "Average_Profit"
    ]
):

    plt.text(
        value,
        bar.get_y() +
        bar.get_height() / 2,
        f" {value:,.0f}",
        va="center"
    )

plt.tight_layout()
plt.show()


# Crop and Season Analysis

crop_season_yield = pd.pivot_table(
    df_clean,
    values="Yield_Tonnes_Ha",
    index="Crop",
    columns="Season",
    aggfunc="mean"
)

crop_season_yield = (
    crop_season_yield
    .reindex(
        columns=final_season_order
    )
)

print(
    "\nAverage Yield by Crop and Season:"
)

print(
    crop_season_yield
    .round(2)
    .to_string()
)


# Visualization 9: Heatmap
# Crop and Season Yield Comparison

plt.figure(figsize=(12, 8))

heatmap_data = (
    crop_season_yield
    .values
)

plt.imshow(
    heatmap_data,
    aspect="auto"
)

plt.colorbar(
    label="Average Yield (Tonnes/Ha)"
)

plt.xticks(
    range(
        len(
            crop_season_yield.columns
        )
    ),
    crop_season_yield.columns,
    rotation=45
)

plt.yticks(
    range(
        len(
            crop_season_yield.index
        )
    ),
    crop_season_yield.index
)

plt.title(
    "Average Yield Across Crops and Seasons"
)

plt.xlabel("Season")
plt.ylabel("Crop")

plt.tight_layout()
plt.show()


# Find Best Crop for Each Season

print(
    "\nBest Performing Crop in Each Season:"
)

for season in final_season_order:

    if season in crop_season_yield.columns:

        values = (
            crop_season_yield[
                season
            ]
            .dropna()
        )

        if len(values) > 0:

            best_crop = (
                values.idxmax()
            )

            best_yield = (
                values.max()
            )

            print(
                f"{season}: "
                f"{best_crop} "
                f"({best_yield:.2f} tonnes/ha)"
            )


# Irrigation Analysis

irrigation_summary = (
    df_clean
    .groupby("Irrigation_Method")
    .agg(
        Average_Yield=(
            "Yield_Tonnes_Ha",
            "mean"
        ),
        Average_Water_Used=(
            "Water_Used_m3",
            "mean"
        ),
        Average_Water_Efficiency=(
            "Water_Efficiency_t_per_1000m3",
            "mean"
        ),
        Average_Profit=(
            "Profit_INR",
            "mean"
        ),
        Records=(
            "Farm_ID",
            "count"
        )
    )
    .sort_values(
        "Average_Yield",
        ascending=False
    )
)

print(
    "\nIrrigation Method Summary:"
)

print(
    irrigation_summary
    .round(2)
    .to_string()
)


# Visualization 10: Horizontal Bar Chart
# Average Yield by Irrigation Method

irrigation_yield_plot = (
    irrigation_summary
    .sort_values(
        "Average_Yield"
    )
)

plt.figure(figsize=(10, 6))

bars = plt.barh(
    irrigation_yield_plot.index,
    irrigation_yield_plot[
        "Average_Yield"
    ]
)

plt.title(
    "Average Yield by Irrigation Method"
)

plt.xlabel(
    "Average Yield (Tonnes/Ha)"
)

plt.ylabel(
    "Irrigation Method"
)

for bar, value in zip(
    bars,
    irrigation_yield_plot[
        "Average_Yield"
    ]
):

    plt.text(
        value,
        bar.get_y() +
        bar.get_height() / 2,
        f" {value:.2f}",
        va="center"
    )

plt.tight_layout()
plt.show()


# Visualization 11: Box Plot
# Yield Variation by Irrigation Method

irrigation_yield_data = []

irrigation_labels = []

for method in (
    df_clean[
        "Irrigation_Method"
    ]
    .dropna()
    .unique()
):

    values = (
        df_clean[
            df_clean[
                "Irrigation_Method"
            ] == method
        ]["Yield_Tonnes_Ha"]
        .dropna()
    )

    if len(values) > 0:

        irrigation_yield_data.append(
            values
        )

        irrigation_labels.append(
            method
        )

plt.figure(figsize=(10, 6))

plt.boxplot(
    irrigation_yield_data,
    labels=irrigation_labels
)

plt.title(
    "Yield Variation by Irrigation Method"
)

plt.xlabel(
    "Irrigation Method"
)

plt.ylabel(
    "Yield (Tonnes/Ha)"
)

plt.xticks(rotation=20)

plt.tight_layout()
plt.show()


# Environmental Analysis

environmental_variables = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Soil_Moisture_pct",
    "Sunlight_Hours_Day"
]

print(
    "\nEnvironmental Variables and Yield Correlation:"
)

environmental_correlations = {}

for variable in environmental_variables:

    correlation = (
        df_clean[
            [
                variable,
                "Yield_Tonnes_Ha"
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    environmental_correlations[
        variable
    ] = correlation

    print(
        f"{variable}: "
        f"{correlation:.3f}"
    )


# Visualization 12: Scatter Plot
# Rainfall vs Yield

rainfall_data = (
    df_clean[
        [
            "Rainfall_mm",
            "Yield_Tonnes_Ha"
        ]
    ]
    .dropna()
)

rainfall_corr = (
    rainfall_data[
        "Rainfall_mm"
    ]
    .corr(
        rainfall_data[
            "Yield_Tonnes_Ha"
        ]
    )
)

print(
    "\nRainfall and Yield Correlation:",
    round(rainfall_corr, 3)
)

plt.figure(figsize=(10, 6))

plt.scatter(
    rainfall_data[
        "Rainfall_mm"
    ],
    rainfall_data[
        "Yield_Tonnes_Ha"
    ],
    alpha=0.5
)

x = rainfall_data[
    "Rainfall_mm"
].values

y = rainfall_data[
    "Yield_Tonnes_Ha"
].values

slope, intercept = np.polyfit(
    x,
    y,
    1
)

x_line = np.linspace(
    x.min(),
    x.max(),
    100
)

y_line = (
    slope * x_line +
    intercept
)

plt.plot(
    x_line,
    y_line,
    linestyle="--"
)

plt.title(
    "Relationship Between Rainfall and Yield"
)

plt.xlabel(
    "Rainfall (mm)"
)

plt.ylabel(
    "Yield (Tonnes/Ha)"
)

plt.tight_layout()
plt.show()


# Visualization 13: Scatter Plot
# Soil Moisture vs Yield

soil_data = (
    df_clean[
        [
            "Soil_Moisture_pct",
            "Yield_Tonnes_Ha"
        ]
    ]
    .dropna()
)

soil_corr = (
    soil_data[
        "Soil_Moisture_pct"
    ]
    .corr(
        soil_data[
            "Yield_Tonnes_Ha"
        ]
    )
)

print(
    "\nSoil Moisture and Yield Correlation:",
    round(soil_corr, 3)
)

plt.figure(figsize=(10, 6))

plt.scatter(
    soil_data[
        "Soil_Moisture_pct"
    ],
    soil_data[
        "Yield_Tonnes_Ha"
    ],
    alpha=0.5
)

x = soil_data[
    "Soil_Moisture_pct"
].values

y = soil_data[
    "Yield_Tonnes_Ha"
].values

slope, intercept = np.polyfit(
    x,
    y,
    1
)

x_line = np.linspace(
    x.min(),
    x.max(),
    100
)

y_line = (
    slope * x_line +
    intercept
)

plt.plot(
    x_line,
    y_line,
    linestyle="--"
)

plt.title(
    "Relationship Between Soil Moisture and Yield"
)

plt.xlabel(
    "Soil Moisture (%)"
)

plt.ylabel(
    "Yield (Tonnes/Ha)"
)

plt.tight_layout()
plt.show()


# State Analysis

state_yield = (
    df_clean
    .groupby("State")[
        "Yield_Tonnes_Ha"
    ]
    .mean()
    .sort_values(
        ascending=False
    )
)

print(
    "\nAverage Yield by State:"
)

print(
    state_yield
    .round(2)
    .to_string()
)


# Visualization 14: Horizontal Bar Chart
# Average Yield by State

state_plot = (
    state_yield
    .sort_values()
)

plt.figure(figsize=(12, 8))

bars = plt.barh(
    state_plot.index,
    state_plot.values
)

plt.title(
    "Average Agricultural Yield by State"
)

plt.xlabel(
    "Average Yield (Tonnes/Ha)"
)

plt.ylabel("State")

for bar, value in zip(
    bars,
    state_plot.values
):

    plt.text(
        value,
        bar.get_y() +
        bar.get_height() / 2,
        f" {value:.2f}",
        va="center"
    )

plt.tight_layout()
plt.show()


# State and Season Analysis

state_season_yield = pd.pivot_table(
    df_clean,
    values="Yield_Tonnes_Ha",
    index="State",
    columns="Season",
    aggfunc="mean"
)

state_season_yield = (
    state_season_yield
    .reindex(
        columns=final_season_order
    )
)

print(
    "\nAverage Yield by State and Season:"
)

print(
    state_season_yield
    .round(2)
    .to_string()
)


# Correlation Analysis

correlation_columns = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Sunlight_Hours_Day",
    "Soil_pH",
    "Soil_Moisture_pct",
    "Nitrogen_kg_ha",
    "Phosphorus_kg_ha",
    "Potassium_kg_ha",
    "Fertilizer_kg_ha",
    "Pesticide_Litre_ha",
    "Seed_Quality_Score",
    "Water_Used_m3",
    "Water_Efficiency_t_per_1000m3",
    "Disease_Pest_Risk_pct",
    "Yield_Tonnes_Ha"
]

correlation_matrix = (
    df_clean[
        correlation_columns
    ]
    .corr()
)

print(
    "\nCorrelation Matrix:"
)

print(
    correlation_matrix
    .round(2)
    .to_string()
)


# Yield Correlation Ranking

yield_correlations = (
    correlation_matrix[
        "Yield_Tonnes_Ha"
    ]
    .drop(
        "Yield_Tonnes_Ha"
    )
    .sort_values(
        ascending=False
    )
)

print(
    "\nVariables Most Positively Correlated "
    "with Yield:"
)

print(
    yield_correlations
    .head(5)
    .round(3)
    .to_string()
)

print(
    "\nVariables Most Negatively Correlated "
    "with Yield:"
)

print(
    yield_correlations
    .tail(5)
    .round(3)
    .to_string()
)


# Statistical Analysis

print(
    "\nStatistical Analysis"
)


# ANOVA: Yield Across Seasons

season_groups = []

for season in final_season_order:

    values = (
        df_clean[
            df_clean["Season"] == season
        ]["Yield_Tonnes_Ha"]
        .dropna()
    )

    if len(values) > 0:

        season_groups.append(
            values.values
        )

if len(season_groups) >= 2:

    f_stat, p_value = stats.f_oneway(
        *season_groups
    )

    print(
        "\nOne-Way ANOVA for Yield Across Seasons:"
    )

    print(
        f"F-statistic: {f_stat:.4f}"
    )

    print(
        f"P-value: {p_value:.6f}"
    )

    if p_value < 0.05:

        print(
            "Result: Seasonal yield differences "
            "are statistically significant."
        )

    else:

        print(
            "Result: Seasonal yield differences "
            "are not statistically significant."
        )


# ANOVA: Yield Across Irrigation Methods

irrigation_groups = []

for method, group in (
    df_clean
    .groupby("Irrigation_Method")
):

    values = (
        group[
            "Yield_Tonnes_Ha"
        ]
        .dropna()
    )

    if len(values) > 0:

        irrigation_groups.append(
            values.values
        )

if len(irrigation_groups) >= 2:

    f_stat_irrigation, p_value_irrigation = (
        stats.f_oneway(
            *irrigation_groups
        )
    )

    print(
        "\nOne-Way ANOVA for Yield Across "
        "Irrigation Methods:"
    )

    print(
        f"F-statistic: "
        f"{f_stat_irrigation:.4f}"
    )

    print(
        f"P-value: "
        f"{p_value_irrigation:.6f}"
    )

    if p_value_irrigation < 0.05:

        print(
            "Result: Yield differences across "
            "irrigation methods are statistically "
            "significant."
        )

    else:

        print(
            "Result: Yield differences across "
            "irrigation methods are not "
            "statistically significant."
        )


# Yield Outlier Analysis

yield_values = (
    df_clean[
        "Yield_Tonnes_Ha"
    ]
    .dropna()
)

Q1 = yield_values.quantile(
    0.25
)

Q3 = yield_values.quantile(
    0.75
)

IQR = Q3 - Q1

lower_limit = (
    Q1 - 1.5 * IQR
)

upper_limit = (
    Q3 + 1.5 * IQR
)

yield_outliers = df_clean[
    (
        df_clean[
            "Yield_Tonnes_Ha"
        ] < lower_limit
    )
    |
    (
        df_clean[
            "Yield_Tonnes_Ha"
        ] > upper_limit
    )
]

print(
    "\nYield Outlier Analysis:"
)

print(
    f"Q1: {Q1:.2f}"
)

print(
    f"Q3: {Q3:.2f}"
)

print(
    f"IQR: {IQR:.2f}"
)

print(
    f"Lower Bound: {lower_limit:.2f}"
)

print(
    f"Upper Bound: {upper_limit:.2f}"
)

print(
    "Number of Yield Outliers:",
    len(yield_outliers)
)


# Find Important Results

best_yield_season = (
    season_yield.idxmax()
)

best_yield_value = (
    season_yield.max()
)

lowest_yield_season = (
    season_yield.idxmin()
)

lowest_yield_value = (
    season_yield.min()
)

best_profit_season = (
    economic_summary[
        "Average_Profit"
    ]
    .idxmax()
)

best_profit_value = (
    economic_summary[
        "Average_Profit"
    ]
    .max()
)

best_crop = (
    crop_summary[
        "Average_Yield"
    ]
    .idxmax()
)

best_crop_yield = (
    crop_summary[
        "Average_Yield"
    ]
    .max()
)

best_irrigation = (
    irrigation_summary[
        "Average_Yield"
    ]
    .idxmax()
)

best_irrigation_yield = (
    irrigation_summary[
        "Average_Yield"
    ]
    .max()
)

best_state = (
    state_yield.idxmax()
)

best_state_yield = (
    state_yield.max()
)

strongest_environmental_variable = max(
    environmental_correlations,
    key=lambda x: abs(
        environmental_correlations[x]
    )
)

strongest_environmental_correlation = (
    environmental_correlations[
        strongest_environmental_variable
    ]
)


# Key Findings

print(
    "\nKEY FINDINGS"
)

print(
    "\n1. Highest average yield season:"
)

print(
    f"{best_yield_season} "
    f"with {best_yield_value:.2f} "
    f"tonnes/ha."
)

print(
    "\n2. Lowest average yield season:"
)

print(
    f"{lowest_yield_season} "
    f"with {lowest_yield_value:.2f} "
    f"tonnes/ha."
)

print(
    "\n3. Most profitable season:"
)

print(
    f"{best_profit_season} "
    f"with average profit of "
    f"INR {best_profit_value:,.2f}."
)

print(
    "\n4. Highest-yielding crop:"
)

print(
    f"{best_crop} "
    f"with {best_crop_yield:.2f} "
    f"tonnes/ha."
)

print(
    "\n5. Highest-yielding irrigation method:"
)

print(
    f"{best_irrigation} "
    f"with {best_irrigation_yield:.2f} "
    f"tonnes/ha."
)

print(
    "\n6. Highest-yielding state:"
)

print(
    f"{best_state} "
    f"with {best_state_yield:.2f} "
    f"tonnes/ha."
)

print(
    "\n7. Environmental variable with the "
    "strongest correlation with yield:"
)

print(
    f"{strongest_environmental_variable} "
    f"with correlation "
    f"{strongest_environmental_correlation:.3f}."
)


# Seasonal Planning Recommendations

print(
    "\nDATA-DRIVEN RECOMMENDATIONS"
)

print(
    f"\n1. Seasonal planning should consider "
    f"the observed yield differences, with "
    f"{best_yield_season} showing the highest "
    f"average yield in this dataset."
)

print(
    "\n2. Crop selection should consider "
    "crop-specific performance across "
    "different seasons rather than applying "
    "the same approach to every season."
)

print(
    f"\n3. {best_irrigation} irrigation shows "
    f"the highest average yield in the dataset. "
    f"Its resource usage and economic performance "
    f"should also be considered before making "
    f"planning decisions."
)

print(
    "\n4. Seasonal differences in rainfall, "
    "temperature and soil moisture should "
    "be considered when planning agricultural "
    "activities."
)

print(
    "\n5. Resource usage such as water and "
    "fertilizer should be monitored across "
    "seasons to identify opportunities for "
    "more efficient agricultural practices."
)

print(
    "\n6. Environmental variables with stronger "
    "relationships with yield deserve further "
    "investigation."
)

print(
    "\n7. Statistical results should be considered "
    "along with descriptive analysis before "
    "concluding that observed seasonal "
    "differences are meaningful."
)

print(
    "\n8. The findings represent relationships "
    "observed in the available dataset and "
    "should not automatically be interpreted "
    "as direct cause-and-effect relationships."
)


# Final Project Summary

print(
    "\nFINAL PROJECT SUMMARY"
)

print(
    f"\nTotal agricultural records analyzed: "
    f"{len(df_clean)}"
)

print(
    f"Number of states: "
    f"{df_clean['State'].nunique()}"
)

print(
    f"Number of districts: "
    f"{df_clean['District'].nunique()}"
)

print(
    f"Number of crops: "
    f"{df_clean['Crop'].nunique()}"
)

print(
    f"Number of seasons: "
    f"{df_clean['Season'].nunique()}"
)

print(
    f"Overall average yield: "
    f"{df_clean['Yield_Tonnes_Ha'].mean():.2f} "
    f"tonnes/ha"
)

print(
    f"Overall average profit: "
    f"INR {df_clean['Profit_INR'].mean():,.2f}"
)

print(
    "\nSeasonal Agriculture Performance Analysis "
    "completed successfully."
)