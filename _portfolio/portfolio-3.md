---
title: "Mobility Analysis using Call Detail Records and Location Data"
excerpt: "Mobility analysis is a crucial part of any modern urbanization structure. Call Detail Records provide very useful information about the behavious and movement of mobile phone users. From tracking movement patterns and understanding urban dynamics to disaster response and crisis mapping, CDR's are very crucial in a wide range of contexts. <br/><img src='/images/500x300.png'>"
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
