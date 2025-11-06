"""
AWS Glue ETL Job - Data Ingestion & Preprocessing
Place this in: AWS Glue -> Jobs

This job:
1. Reads airline data from S3
2. Performs data cleaning
3. Engineers features
4. Writes to processed S3 bucket
"""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pysglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import boto3

# Initialize Glue context
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_INPUT_BUCKET', 'S3_OUTPUT_BUCKET'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Parameters
S3_INPUT_PATH = f"s3://{args['S3_INPUT_BUCKET']}/raw/airline_data/"
S3_OUTPUT_PATH = f"s3://{args['S3_OUTPUT_BUCKET']}/processed/features/"


def clean_airline_data(df):
    """
    Clean airline operational data
    """
    # Convert date column
    df = df.withColumn("FlightDate", F.to_date(F.col("FlightDate")))
    
    # Remove critical nulls
    df = df.filter(
        F.col("FlightDate").isNotNull() &
        F.col("Reporting_Airline").isNotNull() &
        F.col("Origin").isNotNull() &
        F.col("Dest").isNotNull()
    )
    
    # Fill numeric nulls
    numeric_cols = ['DepDelay', 'ArrDelay', 'TaxiOut', 'TaxiIn', 'AirTime']
    for col in numeric_cols:
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0)))
    
    # Remove outliers
    df = df.filter(
        (F.col("TaxiOut") >= 0) & (F.col("TaxiOut") <= 300) &
        (F.col("TaxiIn") >= 0) & (F.col("TaxiIn") <= 300) &
        (F.col("DepDelay") >= -120) & (F.col("DepDelay") <= 1440) &
        (F.col("ArrDelay") >= -120) & (F.col("ArrDelay") <= 1440)
    )
    
    # Remove duplicates
    df = df.dropDuplicates([
        'FlightDate', 'Reporting_Airline', 'Tail_Number',
        'Origin', 'Dest', 'CRSDepTime'
    ])
    
    return df


def engineer_features(df):
    """
    Create operational efficiency and delay features
    """
    # Route identifier
    df = df.withColumn("Route", F.concat_ws("-", F.col("Origin"), F.col("Dest")))
    
    # Time components
    df = df.withColumn("Year", F.year(F.col("FlightDate")))
    df = df.withColumn("Month", F.month(F.col("FlightDate")))
    df = df.withColumn("DayOfWeek", F.dayofweek(F.col("FlightDate")))
    
    # Delay indicators
    df = df.withColumn("Is_ArrDelayed_15min", 
                       F.when(F.col("ArrDelay") > 15, 1).otherwise(0))
    
    # Taxi efficiency (deviation from airport median)
    window_origin = Window.partitionBy("Origin")
    window_dest = Window.partitionBy("Dest")
    
    df = df.withColumn("TaxiOut_Airport_Median", 
                       F.percentile_approx(F.col("TaxiOut"), 0.5).over(window_origin))
    df = df.withColumn("TaxiOut_Deviation", 
                       F.col("TaxiOut") - F.col("TaxiOut_Airport_Median"))
    
    df = df.withColumn("TaxiIn_Airport_Median",
                       F.percentile_approx(F.col("TaxiIn"), 0.5).over(window_dest))
    df = df.withColumn("TaxiIn_Deviation",
                       F.col("TaxiIn") - F.col("TaxiIn_Airport_Median"))
    
    # Air time efficiency
    df = df.withColumn("Expected_AirTime", (F.col("Distance") / 500) * 60)
    df = df.withColumn("AirTime_Deviation", 
                       F.col("AirTime") - F.col("Expected_AirTime"))
    
    # Delay recovery
    df = df.withColumn("Delay_Recovery", 
                       F.col("DepDelay") - F.col("ArrDelay"))
    df = df.withColumn("Made_Up_Time",
                       F.when(F.col("Delay_Recovery") > 5, 1).otherwise(0))
    
    # Aircraft rotation features
    window_aircraft = Window.partitionBy("Tail_Number").orderBy("FlightDate", "DepTime")
    
    df = df.withColumn("Prev_Flight_ArrDelay",
                       F.lag(F.col("ArrDelay"), 1).over(window_aircraft))
    df = df.withColumn("Prev_Flight_Dest",
                       F.lag(F.col("Dest"), 1).over(window_aircraft))
    
    df = df.withColumn("Is_Valid_Rotation",
                       F.when(F.col("Prev_Flight_Dest") == F.col("Origin"), 1).otherwise(0))
    
    # Cascade risk indicator
    df = df.withColumn("Is_Cascade_Victim",
                       F.when((F.col("LateAircraftDelay") > 0), 1).otherwise(0))
    
    return df


def calculate_aggregates(df):
    """
    Calculate route and carrier level aggregates
    """
    # Route-level aggregates
    route_agg = df.groupBy("Route").agg(
        F.count("*").alias("Route_Flight_Count"),
        F.mean("ArrDelay").alias("Route_Avg_ArrDelay"),
        F.mean("Is_ArrDelayed_15min").alias("Route_Delay_Rate")
    )
    
    df = df.join(route_agg, on="Route", how="left")
    
    # Carrier-level aggregates
    carrier_agg = df.groupBy("Reporting_Airline").agg(
        F.mean("ArrDelay").alias("Carrier_Avg_ArrDelay"),
        F.mean("Is_ArrDelayed_15min").alias("Carrier_Delay_Rate")
    )
    
    df = df.join(carrier_agg, on="Reporting_Airline", how="left")
    
    return df


# Main ETL pipeline
print("Reading data from S3...")
raw_df = spark.read.parquet(S3_INPUT_PATH)

print("Cleaning data...")
clean_df = clean_airline_data(raw_df)

print("Engineering features...")
features_df = engineer_features(clean_df)

print("Calculating aggregates...")
final_df = calculate_aggregates(features_df)

print(f"Writing processed data to {S3_OUTPUT_PATH}...")
final_df.write.mode("overwrite").partitionBy("Year", "Month").parquet(S3_OUTPUT_PATH)

print("ETL job complete!")
job.commit()
