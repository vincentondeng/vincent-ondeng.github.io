---
title: "Mobility Analysis using Call Detail Records and Location Data"
excerpt: "Mobility analysis is a crucial part of any modern urbanization structure. Call Detail Records provide very useful information about the behavious and movement of mobile phone users. From tracking movement patterns and understanding urban dynamics to disaster response and crisis mapping, CDR's are very crucial in a wide range of contexts. <br/><img src='/images/image-1.png'>"
collection: portfolio
---

During the COVID-19 lockdowns in 2020, governments around the world implemented various strategies to reduce human mobility and limit the spread of the virus. One of the methods used to monitor and enforce movement restrictions was the analysis of cell signals, particularly through Call Detail Records (CDRs) and other forms of mobile network data. These methods offered a way to track population movement patterns at scale while maintaining user anonymity.

Mobile phones connect to nearby cell towers, and each connection leaves a trace that can be used to estimate a user’s location over time. By analyzing these signals, authorities could detect patterns of movement and identify areas where restrictions were being ignored.

In this project, I am going to explore, how this could be possible as well as harness the power of Spark to analyze this large dataset.

### Objectives

1. Use Pyspark to process call records for 1.5 million users.
2. Generate some summary statistics
3. Explore the relationship between average unque locations per user and their average radius of gyration

### Getting Started
#### Setting up the Environment
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, to_timestamp, rand, countDistinct
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from pathlib import Path
import sys
```
The tools used in this exercise include PySpark, which provides distributed data processing capabilities through SparkSession and functions like `col`, `to_date`, and `countDistinct` for efficient transformations and aggregations. Libraries like Pandas and NumPy enable data manipulation and numerical computations, while Seaborn and Matplotlib are used for creating visualizations to explore relationships in the data. Additional utilities like random and Path help with sampling and file management.

#### Setting Global Parameters and Variables
```python
# Setup global parameters and variables
MISC_PROCESSING_PARAMS = {'distance_threshold': 2, 'min_unique_locs': 2,'datetime_col': 'datetime',
                        'userid': 'user_id','x': 'lon', 'y': 'lat'}
```

#### Task 1: Preprocessing the Data Using Spark
The first hurdle in transforming this messy call detail records into a structured formart suitable for analysis. The `preprocess_cdrs_using_spark` below  leverages Spark's distributed computing capabilities to handle large datasets efficiently. Initial operations, such as droping unwanted columns or removing observations with null values such as location or cell id happen in this function.

```python
def preprocess_cdrs_using_spark(file_or_folder=None, number_of_users_to_sample=None,
                                output_csv=None, date_format='yyyyMMddHHmmss',
                                debug_mode=True, loc_file=None, save_to_csv=False):
    """
    In this function, we perfom some basic preprocessing such as below:
    1. rename columns
    2. change some data types
    3. Add location details
    Eventually, we will sample the data to use for our analysis
    :param data_folder:
    :param output_csv_for_sample_users:
    :return:
    """

    spark = SparkSession.builder \
        .appName("CDR Preprocessing") \
        .master("local[2]") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "6g") \
        .config("spark.sql.shuffle.partitions", "16") \
        .getOrCreate()

    # read data with spark
    df = spark.read.option("header", True).csv(file_or_folder)

    # Renaming the last column to "cellID"
    df = df.withColumnRenamed("last calling cellid", "cellId")

    # repartition to speed up
    df = df.repartition(16)

    # Using spark sample function when debug mode is True
    if debug_mode:
        df = df.sample(withReplacement=False, fraction=0.5)

    # Drop the "call duration column"
    df = df.drop("call duration")

    # Renaming columns
    df = df.withColumnRenamed("cdr type", "cdrType").withColumnRenamed("calling phonenumber", "user_id")
    df = df.drop("cdrType")

    # Add date and timestamp columns
    df = df.withColumn("datetime", to_timestamp(col("cdr datetime"), date_format))
    df = df.withColumn("date", to_date(col("datetime")))

    # use spark filter() function to remove null phoneNumbers i.e user_id
    df = df.filter(col("user_id").isNotNull())

    # Load location details using pandas and then joing with the cdr data
    loc_df = pd.read_csv(loc_file)
    loc_df.rename(columns={"cell_id": "cellId"}, inplace=True)
    spark_loc_df = spark.createDataFrame(loc_df)
    df = df.join(spark_loc_df, on="cellId", how="left")

    # Drop rows with null values in critical columns
    critical_columns = ["cellId", "lat", "lon", "site_id"]
    df = df.dropna(subset=critical_columns)

    # This section will select a given number of users if specified
    if number_of_users_to_sample:
        sampled_users_df = (
            df.select("user_id")
            .distinct()
            .orderBy(rand())
            .limit(number_of_users_to_sample)
        )
        df = df.join(sampled_users_df, on="user_id", how="inner")

    # Cache the Dataframe and since cache is lazy, show the first 10 observations to trigger cache
    df.cache()
    df.show(10)

    # Clear intermediate data except for the final DataFrame
    for table_name in spark.catalog.listTables():
        spark.catalog.dropTempView(table_name.name)

    for df_var in locals().copy():
        if isinstance(locals()[df_var], type(df)) and locals()[df_var] is not df:
            locals()[df_var].unpersist(blocking=True)
    return df
```
The key aspect of this function is the use of `df.show()` immediately after `df.cache()`. When you call `cache()` on a DataFrame in Spark, it stores the DataFrame in memory for faster access during subsequent operations. However, `cache()` is a lazy operation in Spark, meaning it doesn't execute until an action is triggered. Using `df.show()` ensures that the DataFrame is actually cached, as `show()` is an action that forces Spark to evaluate the DataFrame and store it in memory.

```
+-------------------+--------+--------------+-------------------+------------+--------+----------+----------+
|           user_id | cellId |  cdr datetime|           datetime|       date | site_id|      lat |      lon |
+-------------------+--------+--------------+-------------------+------------+--------+----------+----------+
|1000047383694070656|20753.0 |20180705201327|2018-07-05 20:13:27| 2018-07-05 |    S81 | -7.410591| 28.453148|
|1000047383694070656|20753.0 |20180705201327|2018-07-05 20:13:27| 2018-07-05 |    S81 | -7.410591| 28.453148|
|1000047383694070656|20753.0 |20180704074833|2018-07-04 07:48:33| 2018-07-04 |    S81 | -7.410591| 28.453148|
|1000047383694070656|20753.0 |20180704074833|2018-07-04 07:48:33| 2018-07-04 |    S81 | -7.410591| 28.453148|
|1000047383694070656|20751.0 |20180708171827|2018-07-08 17:18:27| 2018-07-08 |    S81 | -7.410591| 28.453148|
|1000047383694070656|20751.0 |20180708171827|2018-07-08 17:18:27| 2018-07-08 |    S81 | -7.410591| 28.453148|
|1000047383694070656|12212.0 |20180701212724|2018-07-01 21:27:24| 2018-07-01 |     S3 |  -8.5834 |  26.8391 |
|1000047383694070656|20853.0 |20180629150149|2018-06-29 15:01:49| 2018-06-29 |     S9 | -7.311427| 28.279376|
|1000047383694070656|20753.0 |20180709092351|2018-07-09 09:23:51| 2018-07-09 |    S81 | -7.410591| 28.453148|
|1000047383694070656|20753.0 |20180709092351|2018-07-09 09:23:51| 2018-07-09 |    S81 | -7.410591| 28.453148|
+-------------------+--------+--------------+-------------------+------------+--------+----------+----------+
only showing top 10 rows
```
### Task 2: Summary Statistics for User Events
#### Summary Statistics
The `summary_stats_for_user_events` function below is designed to compute various statistics related to the number of calls per user in a Spark DataFrame. It begins by grouping the data by `user_id` and counting the number of calls made by each user.

The function calculates key statistical measures including the mean, median, standard deviation, minimum, maximum, and quantiles (25th, 50th, and 75th percentiles) of the call counts.

The computed statistics are then consolidated into a dictionary(before saving into a csv), with each statistic represented as a key-value pair, making it easy to access and use the results for further analysis.

```python
def summary_stats_for_user_events(df):
    """
    Compute statistics (mean, median, std, min, max, quantiles) for the number of calls per user in parallel using Spark.

    :param df: Spark DataFrame with 'user_id' and 'call_id' columns
    :return: A dictionary with calculated statistics
    """
    # Group by user_id and count the number of calls per user
    user_call_counts = df.groupBy("user_id").agg(count("*").alias("num_calls"))

    # Compute statistics in parallel
    stats = user_call_counts.agg(
        mean("num_calls").alias("mean"),
        expr("percentile_approx(num_calls, 0.5)").alias("median"),
        stddev("num_calls").alias("std"),
        spark_min("num_calls").alias("min"),
        spark_max("num_calls").alias("max"),
        expr("percentile_approx(num_calls, array(0.25, 0.5, 0.75))").alias("quantiles"),
        count("*").alias("num_users")
    ).collect()[0]

    # Extract quantiles
    quantiles = stats["quantiles"]

    # Return results as a dictionary
    return {
        "mean": stats["mean"],
        "median": stats["median"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
        "Q1 (25%)": quantiles[0],
        "Median (50%)": quantiles[1],
        "Q3 (75%)": quantiles[2],
        "num_users": stats["num_users"]  # Include the user count
    }
```
 This is the outcome of the previous function;
 ```
 mean: 7.793482666666667
median: 3
std: 15.807889375792001
min: 1
max: 8277
Q1 (25%): 1
Median (50%): 3
Q3 (75%): 8
num_users: 1500000
 ```

 #### Number of Days
 The `compute_date_range` function below identifies the earliest and latest dates in a Spark DataFrame. It computes these dates by applying aggregation functions to determine the minimum and maximum values in the date column. Once the range is established, the function calculates the number of days between the two dates by subtracting the earliest date from the latest date.

 ```python
 def compute_date_range(df):
  """
  Computes the number of days
  First finds the earliest date
  Second find the latest date
  Calculate the difference
  """
  if not df.is_cached:
    df.cache()

  # Take the column date
  df = df.withColumn("date", df["date"].cast("date"))

  start_date_rdd = df.select(spark_min("date")).rdd
  end_date_rdd = df.select(spark_max("date")).rdd

  start_date = start_date_rdd.collect()[0][0]
  end_date = end_date_rdd.collect()[0][0]

  # Computes the difference
  total_days = (end_date - start_date).days

  return start_date, end_date, total_days
 ```

 ```
 Start Date: 2018-06-29
End Date: 2018-07-13
Total Days in Range: 14
 ```

 ### Task 3: Individual User Mobility Statistics

##### Unique Locations Visited

For each user, the number of unique locations represents how many different places they have visited over a given period. It is calculated as \( L_i \), where \( i \) refers to an individual user. This metric reflects the diversity of a user's movement, with higher values indicating that the user has visited a wider range of distinct locations. It provides insight into how exploratory or localized a person’s mobility behavior is.

##### Mean Radius of Gyration

For each user, the radius of gyration measures how far they typically move from their center of activity or average location. It is calculated using the formula:

\[
r_g^{(i)} = \sqrt{ \frac{1}{n_i} \sum_{j=1}^{n_i} \left\| \vec{l}_{i,j} - \vec{r}_i \right\|^2 }
\]

where \( \vec{l}_{i,j} \) is the location visited at time \( j \), \( \vec{r}_i \) is the user's center of mass, and \( n_i \) is the total number of observations for user \( i \). 

- A small radius of gyration indicates that the person tends to stay near the average location, implying less variability in their movement.
- A larger radius of gyration indicates that the person’s locations are more spread out from the average location, suggesting more extensive movement or higher variability.


The radius of gyration is implemented using a pandas users defined function which is a lot faster that converting the spark dataframe to pandas before doing the computation.


The `generate_basic_user_attributes_with_spark` function computes basic mobility metrics for each user from a Spark DataFrame containing mobility data. 
It extracts relevant columns based on a provided parameter dictionary and calculates two key attributes per day: the number of unique locations visited and the radius of gyration, which reflects how far users typically move from their average location. 

The function uses a custom UDF to compute the radius of gyration and then aggregates daily metrics to generate user-level summaries, including average unique locations per day, average radius of gyration, and the number of active days. Users with fewer active days than a specified threshold can be filtered out, and the resulting attributes are saved to a CSV file.

```python
def generate_basic_user_attributes_with_spark(dfu, output_csv, misc_params, num_events_threshold=None):
    """
    Generate basic user mobility attributes using Spark and save the results to a CSV file.

    :param dfu: Spark DataFrame with user mobility data.
    :param output_csv: Path to save the resulting CSV file.
    :param misc_params: Dictionary containing column names and parameters.
    :param num_events_threshold: Minimum number of events required to process a user.
    :return: Spark DataFrame with user mobility attributes.
    """
    if not dfu.is_cached:
        dfu.cache()

    # Extract column names and parameters from misc_params
    user_id_col = misc_params["userid"]
    lat_col = misc_params["y"]
    lon_col = misc_params["x"]
    datetime_col = misc_params["datetime_col"]
    min_unique_locs = misc_params["min_unique_locs"]

    # Step 1: Calculate unique locations and radius of gyration per day
    daily_mobility = (
        dfu.groupBy(user_id_col, "date")
        .agg(
            countDistinct(lat_col, lon_col).alias("unique_locs"),
            collect_list(lat_col).alias("lat_list"),
            collect_list(lon_col).alias("lon_list"),
        )
    )

    # Define UDF for radius of gyration
    def calculate_radius_of_gyration(lat_list, lon_list):
        if len(lat_list) <= 1:
            return 0.0
        coords = np.array(list(zip(lat_list, lon_list)))
        centroid = coords.mean(axis=0)
        distances = np.sqrt(np.sum((coords - centroid) ** 2, axis=1))
        return np.sqrt(np.mean(distances ** 2))

    from pyspark.sql.functions import pandas_udf
    @pandas_udf(DoubleType())
    def radius_of_gyration_udf(lat_list, lon_list):
        return pd.Series(
            [calculate_radius_of_gyration(lat, lon) for lat, lon in zip(lat_list, lon_list)]
        )

    # Add radius of gyration column
    daily_mobility = daily_mobility.withColumn(
        "radius_of_gyration", radius_of_gyration_udf(col("lat_list"), col("lon_list"))
    )

    # Aggregate by user to compute overall metrics
    user_mobility = (
        daily_mobility.groupBy(user_id_col)
        .agg(
            countDistinct("date").alias("active_days"),
            avg("unique_locs").alias("avg_unique_locs_per_day"),
            avg("radius_of_gyration").alias("avg_radius_of_gyration"),
        )
    )

    # Apply the threshold filter
    if num_events_threshold:
        user_mobility = user_mobility.filter(col("active_days") >= num_events_threshold)

    # Save the Output to CSV
    user_mobility.write.csv(output_csv, header=True, mode="overwrite")
    print(f"User mobility attributes saved to: {output_csv}")

    return user_mobility
```

### Task 4: Mean Radius of Gyration vs Average Locations per day

The `explore_correlation_between_metrics` function analyzes the relationship between users’ average radius of gyration (`avg_radius_of_gyration`) and their average number of unique locations per day (`avg_unique_locs_per_day`) using a Spark DataFrame. 

This function utilizes the `corr` function in sparksql to calculate the correlation between average radius of gyration per day and the average number of locations per user.

```python
def explore_correlation_between_metrics(user_mobility_df, output_plot="correlation_plot.png"):
    """
    Explore the correlation between avg_Rg and avg_locs_per_day.

    :param user_mobility_df: Spark DataFrame containing 'avg_radius_of_gyration' and 'avg_unique_locs_per_day'.
    :param output_plot: Path to save the correlation plot.
    :return: Correlation coefficient.
    """
    if not user_mobility_df.is_cached:
        user_mobility_df.cache()

    # Calculate correlation using Spark
    correlation = user_mobility_df.select(
        corr("avg_radius_of_gyration", "avg_unique_locs_per_day").alias("correlation")
    ).collect()[0]["correlation"]

    print(f"Correlation coefficient between avg_Rg and avg_locs_per_day: {correlation:.4f}")

    # Convert Spark DataFrame to Pandas for visualization
    pdf = user_mobility_df.select("avg_radius_of_gyration", "avg_unique_locs_per_day").toPandas()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pdf, x="avg_radius_of_gyration", y="avg_unique_locs_per_day")
    sns.regplot(
        data=pdf,
        x="avg_radius_of_gyration",
        y="avg_unique_locs_per_day",
        scatter=False,
        color="red",
    )
    plt.title("Correlation Between Avg Radius of Gyration and Avg Locations Per Day")
    plt.xlabel("Average Radius of Gyration (avg_Rg)")
    plt.ylabel("Average Locations Per Day (avg_locs_per_day)")
    plt.grid()
    plt.savefig(output_plot)
    plt.show()

    print(f"Correlation plot saved to: {output_plot}")

    return correlation
```

```
Correlation coefficient between avg_Rg and avg_locs_per_day: 0.3235
```
![alt text](/images/image-1.png)

### Strategies
Strategies such as repartitioning the data ensured balanced load distribution across partitions, while persisting intermediate results (`persist()` and `cache()`) avoided redundant computations during repeated access. During debugging, sampling smaller subsets of the data helped speed up testing without compromising scalability for full-scale processing. 
Additionally, optimizing complex metric calculations, such as the radius of gyration, involved leveraging built-in Spark SQL functions and efficient vectorized operations through Pandas UDFs. Grouping data early in the workflow reduced redundant processing, and unnecessary data format conversions (e.g., from Spark to Pandas) to maintain efficiency for the large dataset. These combined strategies ensured that both large-scale data handling and computationally intensive tasks were addressed effectively.

The complete Jupyter Notebook for this exercise is available at the following link:
