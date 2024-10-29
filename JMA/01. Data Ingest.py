# Databricks notebook source
!pip install xmltodict 
!pip install llama-index-core~=0.10.59 llama-index-embeddings-text-embeddings-inference~=0.1.4 llama-index-llms-openai-like~=0.1.3 msal~=1.28.0 openai~=1.30.1 msal~=1.28.0 "unstructured[pdf,docx]" transformers
!pip install mlflow
!pip install databricks-vectorsearch==0.22
!pip install flashrank==0.2.0 langchain==0.1.16 databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run /Workspace/Repos/nchandra@munichre.com/jma_rag/JMA/utils/_helper_functions

# COMMAND ----------

import yaml
import os
import glob
import numpy as np
import pprint as pp
import time
import requests
import xmltodict
import pandas as pd
import math
import logging
import sys
from pathlib import Path
from datetime import datetime

import os
import sys
sys.path.append('../utils')
import nest_asyncio

nest_asyncio.apply()

# Load parameters from YAML file
yaml_file = open("utils/parameters.yaml", "r")
variables = yaml.safe_load(yaml_file)

# COMMAND ----------

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader, Document
# from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor, TitleExtractor, KeywordExtractor
from llama_index.core.schema import MetadataMode

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

# COMMAND ----------

# MAGIC %md
# MAGIC # 01. Set up

# COMMAND ----------

catalog = variables['CATALOG']
Schema = variables['SCHEMA']
volume = variables['VOLUME']
directory = variables['DIRECTORY']
table_name = variables['TABLE_NAME']
index_name = variables['INDEX_NAME']
vs_endpoint_name = variables['VS_ENDPOINT_NAME']
xml_store_path = f"/Volumes/{catalog}/{Schema}/project/{directory}/"

# COMMAND ----------

# When TEST is True, no data is saved
TEST = True
# Operation Mode ("live", "test_files", "test_urls")
# live: for downloading xml data from JMA at regular interval
# test_files: if you have JMA xml files already saved in ADLS and you want to process it
# test_urls: if you have urls to JMA xml files and you want to process it
mode = "live"

# COMMAND ----------

# on test_urls mode, please specify the urls of the xml in a list format
test_event_urls = [
    "https://www.data.jma.go.jp/developer/xml/data/20240814010654_0_VPTW60_010000.xml",
    "https://www.data.jma.go.jp/developer/xml/data/20240813220252_0_VPTW60_010000.xml",
    "https://www.data.jma.go.jp/developer/xml/data/20240813190014_0_VPTW60_010000.xml",
    "https://www.data.jma.go.jp/developer/xml/data/20240813160042_0_VPTW60_010000.xml",
    "https://www.data.jma.go.jp/developer/xml/data/20240814004927_0_VPTW62_010000.xml",
    "https://www.data.jma.go.jp/developer/xml/data/20240813214051_0_VPTW62_010000.xml",
    "https://www.data.jma.go.jp/developer/xml/data/20240813184515_0_VPTW62_010000.xml",
    "https://www.data.jma.go.jp/developer/xml/data/20240813154050_0_VPTW62_010000.xml",
]

# COMMAND ----------

# for test_files mode, please specify the path to where xml files are saved
# TC212
test_xml_path = "/Volumes/workspace_us_east_2/default/project/jma_data/"

# COMMAND ----------

# MAGIC %md
# MAGIC # 02. Parameters

# COMMAND ----------

# Get URL of XML feed
url = "https://www.data.jma.go.jp/developer/xml/feed/extra.xml"
response = requests.get(url)

# Parse the XML feed to a dict
xml = xmltodict.parse(response.content)

# Feed Entry
entries = xml["feed"]["entry"]

# COMMAND ----------

# MAGIC %md
# MAGIC # 03. Read in event log and set default dataframe

# COMMAND ----------

dtype_schema = {
    "EventID": "object",
    "InfoType": "object",
    "Serial": "int32",
    "InfoKindVersion": "object",
    "TyphoonName": "object",
    "CenterLat": "float64",
    "CenterLon": "float64",
    "CenterLocation": "object",
    "CenterMovementDirection": "object",
    "CenterSpeed": "float64",
    "CentralPressure": "float64",
    "MaxWindspeedNearCenter": "float64",
    "MaxGustSpeed": "float64",
    "Bofuuiki_Direction1": "object",
    "Bofuuiki_Radius1": "float64",
    "Bofuuiki_Direction2": "object",
    "Bofuuiki_Radius2": "float64",
    "Kyofuuiki_Direction1": "object",
    "Kyofuuiki_Radius1": "float64",
    "Kyofuuiki_Direction2": "object",
    "Kyofuuiki_Radius2": "float64",
    "URL": "object",
}

# COMMAND ----------

# Define DataFrame structure with default values
columns = (
    list(dtype_schema.keys())[:4] + ["ObservationTime"] + list(dtype_schema.keys())[4:]
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 04. Collect new events from JMA XML

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Helper Functions

# COMMAND ----------

def process_meteorological_info(row, meteorological_info):
    """Extracts and processes meteorological information."""
    items = meteorological_info["Item"]
    items = items if isinstance(items, list) else [items]

    for item in items:
        kinds = item["Kind"]
        kinds = kinds if isinstance(kinds, list) else [kinds]

        for kind in kinds:
            prop = kind["Property"]
            if prop["Type"] == "呼称" and "TyphoonNamePart" in prop:
                row["TyphoonName"] = prop["TyphoonNamePart"]["Name"]

            # Extract center info
            if prop["Type"] == "中心":
                process_center_info(row, prop)

            # Extract wind info
            if prop["Type"] == "風":
                process_wind_info(row, prop)

# COMMAND ----------

def process_center_info(row, prop):
    """Processes and extracts center information."""
    center_coordinate = prop["CenterPart"]["jmx_eb:Coordinate"][1]["#text"]
    row["CenterLat"] = "{:.2f}".format(int(center_coordinate[1:5]) / 100)
    row["CenterLon"] = "{:.2f}".format(int(center_coordinate[6:11]) / 100)
    row["CenterLocation"] = prop["CenterPart"]["Location"]
    row["CenterMovementDirection"] = (
        prop["CenterPart"].get("jmx_eb:Direction", {}).get("#text", np.NaN)
    )
    row["CenterSpeed"] = (
        prop["CenterPart"].get("jmx_eb:Speed", [{}])[1].get("#text", np.NaN)
    )
    row["CentralPressure"] = (
        prop["CenterPart"].get("jmx_eb:Pressure", {}).get("#text", np.NaN)
    )

# COMMAND ----------

def process_wind_info(row, prop):
    """Processes and extracts wind information."""
    row["MaxWindspeedNearCenter"] = (
        prop["WindPart"].get("jmx_eb:WindSpeed", [{}])[1].get("#text", np.NaN)
    )
    row["MaxGustSpeed"] = (
        prop["WindPart"].get("jmx_eb:WindSpeed", [{}])[3].get("#text", np.NaN)
    )

    # Process 暴風域 data
    process_area_info(row, prop, "WarningAreaPart", "暴風域", "Bofuuiki")
    # Process 強風域 data
    process_area_info(row, prop, "WarningAreaPart", "強風域", "Kyofuuiki")

# COMMAND ----------

def process_area_info(row, prop, part_name, area_type, prefix):
    """Processes and extracts area information."""
    try:
        # Find the specific area part based on the type
        area_part = next(item for item in prop[part_name] if item["@type"] == area_type)

        # Access the 'jmx_eb:Axis' elements within the 'jmx_eb:Circle' structure
        axes = area_part["jmx_eb:Circle"]["jmx_eb:Axes"]["jmx_eb:Axis"]

        # Initialize a list to store direction and radius pairs
        direction_radius_pairs = []

        # Ensure axes is a list for uniform processing
        if not isinstance(axes, list):
            axes = [axes]

        # Process each axis to extract direction and radius
        for axis in axes:
            # Extract the direction, preferring '#text' over '@description'
            direction = axis["jmx_eb:Direction"].get(
                "#text", axis["jmx_eb:Direction"].get("@description")
            )
            # get the second radius in km
            radius = axis["jmx_eb:Radius"][1]["#text"]

            if axis["jmx_eb:Direction"]["@unit"] == "８方位漢字" and (
                axis["jmx_eb:Direction"].get("@type") == "方向"
                or axis["jmx_eb:Direction"].get("@description") == "全域"
            ):
                direction_radius_pairs.append((direction, radius))

        if direction_radius_pairs:
            (
                row[f"{prefix}_Direction1"],
                row[f"{prefix}_Radius1"],
            ) = direction_radius_pairs[0]
        if len(direction_radius_pairs) > 1:
            (
                row[f"{prefix}_Direction2"],
                row[f"{prefix}_Radius2"],
            ) = direction_radius_pairs[1]

    except (KeyError, IndexError, TypeError) as e:
        # logger.debug(f"Could not process {area_type}: {e}")
        row[f"{prefix}_Direction1"] = np.NaN
        row[f"{prefix}_Radius1"] = np.NaN
        row[f"{prefix}_Direction2"] = np.NaN
        row[f"{prefix}_Radius2"] = np.NaN

# COMMAND ----------

def save_event_xml(content, filename, xml_store_path, event_year, event_id):
    """Saves the event XML file to ADLS and verifies the move."""
    try:
        temp_path = Path("/dbfs/FileStore/")
        temp_path.mkdir(parents=True, exist_ok=True)
        temp_save_path = temp_path / filename

        with open(temp_save_path, "wb") as file:
            file.write(content)

        final_destination_subfolder = os.path.join(
            "dbfs:/mnt/data/", xml_store_path, event_year, event_id
        )
        dbutils.fs.mkdirs(final_destination_subfolder)
        dbutils.fs.mv(
            f"dbfs:/FileStore/{filename}",
            os.path.join(final_destination_subfolder, filename),
        )
        # logger.info(f"File moved successfully to {final_destination_subfolder}")
        return True
    except Exception as save_err:
        # logger.error(f"Error saving or moving file: {save_err}")
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Main Function

# COMMAND ----------

def read_event_xml(
    xml_source, new_link, event_dict, xml_store_path, filename, save_xml=True
):
    try:
        sub_xml = None
        # Read the event's XML
        if xml_source == "url":
            # Read the event's XML
            try:
                sub_response = requests.get(new_link)

                if sub_response.status_code != 200:
                    print(
                        f"Failed to fetch URL: {new_link} with status code {sub_response.status_code}"
                    )
                    return event_dict  # Move to the next URL

                sub_xml = xmltodict.parse(sub_response.content)
            except requests.exceptions.RequestException as e:
                # Handle network-related errors
                print(f"Network error occurred: {e}")
                return event_dict  # Move to the next URL

        else:
            # Open and read the XML file
            with open(new_link, "r", encoding="utf-8") as file:
                sub_response = file.read()
                sub_xml = xmltodict.parse(sub_response)

        # Proceed only if sub_xml was successfully assigned
        if sub_xml is None:
            print(f"Failed to parse XML from {new_link}")
            return event_dict

        event_id = sub_xml["Report"]["Head"]["EventID"]

        meteorological_infos = sub_xml["Report"]["Body"]["MeteorologicalInfos"]
        meteorological_info_list = (
            meteorological_infos["MeteorologicalInfo"]
            if isinstance(meteorological_infos["MeteorologicalInfo"], list)
            else [meteorological_infos["MeteorologicalInfo"]]
        )

        for meteorological_info in meteorological_info_list:
            datetime_type = meteorological_info["DateTime"]["@type"]

            if datetime_type == "実況":
                observation_time = meteorological_info["DateTime"]["#text"]
                event_year = observation_time[:4]

                # Initialize row with default values
                row = {
                    "EventID": event_id,
                    "ObservationTime": observation_time,
                    "InfoType": np.NaN,
                    "Serial": np.NaN,
                    "InfoKindVersion": np.NaN,
                    "TyphoonName": np.NaN,
                    "CenterLat": np.NaN,
                    "CenterLon": np.NaN,
                    "CenterLocation": np.NaN,
                    "CenterMovementDirection": np.NaN,
                    "CenterSpeed": np.NaN,
                    "CentralPressure": np.NaN,
                    "MaxWindspeedNearCenter": np.NaN,
                    "MaxGustSpeed": np.NaN,
                    "Bofuuiki_Direction1": np.NaN,
                    "Bofuuiki_Radius1": np.NaN,
                    "Bofuuiki_Direction2": np.NaN,
                    "Bofuuiki_Radius2": np.NaN,
                    "Kyofuuiki_Direction1": np.NaN,
                    "Kyofuuiki_Radius1": np.NaN,
                    "Kyofuuiki_Direction2": np.NaN,
                    "Kyofuuiki_Radius2": np.NaN,
                    "URL": np.NaN,
                }

                # Extract information
                try:
                    row["URL"] = new_link
                    row["InfoType"] = sub_xml["Report"]["Head"]["InfoType"]
                    row["Serial"] = sub_xml["Report"]["Head"]["Serial"]
                    row["InfoKindVersion"] = sub_xml["Report"]["Head"][
                        "InfoKindVersion"
                    ]

                    process_meteorological_info(row, meteorological_info)
                except KeyError as e:
                    # logger.error(f"Missing key while processing event {event_id}: {e}")
                    print(f"Missing key while processing event {event_id}: {e}")

                # Save event xml to Catalyst ADLS
                if save_xml:
                    save_event_xml(
                        sub_response.content,
                        filename,
                        xml_store_path,
                        event_year,
                        event_id,
                    )

                # Create a unique key based on EventID and ObservationTime
                unique_key = (event_id, observation_time)
                # Store row in dictionary
                event_dict[unique_key] = row

        return event_dict

    except requests.RequestException as req_err:
        # logger.error(f"Request error: {req_err}")
        print(f"Request error: {req_err}")
    except Exception as general_err:
        # logger.error(f"General error: {general_err}")
        print(f"General error: {general_err}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Run Event Collection and Processing

# COMMAND ----------

def process_test_urls(test_event_urls, event_dict, xml_store_path, save_xml=False):
    for new_link in test_event_urls:
        filename = Path(new_link).name
        print("----------------------------------------------------")
        print(new_link)
        time.sleep(1)
        event_dict = read_event_xml(
            "url", new_link, event_dict, xml_store_path, filename, save_xml
        )
    return event_dict

# COMMAND ----------

def process_test_xml_files(xml_files, event_dict, xml_store_path, save_xml=False):
    for xml in xml_files:
        filename = Path(xml).name
        print("----------------------------------------------------")
        print(xml)
        time.sleep(1)

        event_dict = read_event_xml(
            "folder", xml, event_dict, xml_store_path, filename, save_xml
        )
    return event_dict

# COMMAND ----------

def live_monitoring(xml, event_dict, xml_store_path, save_xml=False):
    typhoon_event_indicator = 0
    for event in xml["feed"]["entry"]:
        if "台風解析・予報情報（５日予報）" in event["title"]:
            typhoon_event_indicator += 1
            new_link = event["link"]["@href"]
            filename = Path(new_link).name
            print("----------------------------------------------------")
            print(new_link)
            time.sleep(1)
            event_dict = read_event_xml(
                "url", new_link, event_dict, xml_store_path, filename, save_xml
            )
    return event_dict

# COMMAND ----------

operation_mode = {
    "live": live_monitoring,
    "test_urls": process_test_urls,
    "test_files": process_test_xml_files,
}

# COMMAND ----------

# Function to execute the selected operation mode
def run_operation_mode(mode, *args):
    if mode in operation_mode:
        # Call the function associated with the selected mode
        return operation_mode[mode](*args)
    else:
        raise ValueError(f"Invalid mode: {mode}")

# COMMAND ----------

# Instantiate event_dict
event_dict = {}

# in TEST mode, do not save the xml and the log
if TEST:
    save_xml = False

else:
    save_xml = True

# Run operation depending on selected mode
if mode == "live":
    event_dict = run_operation_mode(mode, xml, event_dict, xml_store_path, save_xml)
elif mode == "test_urls":
    event_dict = run_operation_mode(
        mode, test_event_urls, event_dict, xml_store_path, save_xml
    )
elif mode == "test_files":
    xml_files = glob.glob(f"{test_xml_path}/*.xml")
    # xml_files
    event_dict = run_operation_mode(
        mode, xml_files, event_dict, xml_store_path, save_xml
    )
else:
    dbutils.notebook.exit("No mode selected")

# COMMAND ----------

event_dict

# COMMAND ----------

# MAGIC %md
# MAGIC # 05. Prepare Data as Documents or Nodes

# COMMAND ----------

from datetime import datetime
documents = []

# Function to handle NaN values
def safe_nan_check(value):
    if value is None or (isinstance(value, float) and value != value):  # Check for NaN
        return "N/A"
    return value

# Iterate over each event ID and its associated attributes
for (event_id, observation_time), attributes in event_dict.items():
    print(f"Event ID: {event_id}, Observation Time: {observation_time}")  # Debug print
    print(f"Attributes: {attributes}")  # Print attributes for debugging
    
    try:
        parsed_time = datetime.fromisoformat(observation_time)
        observation_date = parsed_time.date()
        observation_time_only = parsed_time.time()

    except ValueError:
        # Handle parsing error if observation_time is not in expected format
        observation_date = "Unknown"
        observation_time_only = "Unknown"

    # Use the safe_nan_check function to handle NaN and None values
    center_speed = safe_nan_check(attributes['CenterSpeed'])
    typhoon_name = attributes['TyphoonName'] if attributes['TyphoonName'] else "Unknown"

    # Create a Document instance
    doc = Document(
        text=(
            f"Event ID: {event_id}\n"
            f"Observation Time: {observation_time}\n"
            f"Info Type: {attributes['InfoType']}\n"
            f"Serial: {attributes['Serial']}\n"
            f"Info Kind Version: {attributes['InfoKindVersion']}\n"
            f"Typhoon Name: {typhoon_name}\n"
            f"Center Latitude: {attributes['CenterLat']}\n"
            f"Center Longitude: {attributes['CenterLon']}\n"
            f"Center Location: {attributes['CenterLocation']}\n"
            f"Center Movement Direction: {attributes['CenterMovementDirection']}\n"
            f"Center Speed: {center_speed} km/h\n"
            f"Central Pressure: {attributes['CentralPressure']} hPa\n"
            f"Max Windspeed Near Center: {attributes['MaxWindspeedNearCenter']} km/h\n"
            f"Max Gust Speed: {attributes['MaxGustSpeed']} km/h\n"
            f"Bofuuiki Direction 1: {attributes['Bofuuiki_Direction1']}, Radius 1: {attributes['Bofuuiki_Radius1']}\n"
            f"Bofuuiki Direction 2: {attributes['Bofuuiki_Direction2']}, Radius 2: {attributes['Bofuuiki_Radius2']}\n"
            f"Kyofuuiki Direction 1: {attributes['Kyofuuiki_Direction1']}, Radius 1: {attributes['Kyofuuiki_Radius1']}\n"
            f"Kyofuuiki Direction 2: {attributes['Kyofuuiki_Direction2']}, Radius 2: {attributes['Kyofuuiki_Radius2']}"
        ),
        metadata={
            "EventID": event_id,
            "TyphoonName": typhoon_name,
            "ObservationDate": str(observation_date),
            "ObservationTime": str(observation_time_only),
            "ObservationDateTime": str(observation_time),
            "InfoType": attributes["InfoType"],
            "Serial": attributes["Serial"],
            "InfoKindVersion": attributes["InfoKindVersion"],
            "TyphoonName": typhoon_name,
        }
    )
    
    # Append the Document to the list
    documents.append(doc)


# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Creating Spark Dataframe and Delta Table

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

# Define the schema for your DataFrame
schema = StructType([
    StructField("id_", StringType(), True),
    StructField("EventID", StringType(), True),
    StructField("TyphoonName", StringType(), True),
    StructField("ObservationDate", StringType(), True),
    StructField("ObservationTime", StringType(), True),
    StructField("ObservationDateTime", StringType(), True),
    StructField("InfoType", StringType(), True),
    StructField("Serial", StringType(), True),
    StructField("InfoKindVersion", StringType(), True),
    StructField("Text", StringType(), True),
    StructField("embedding", ArrayType(FloatType(), True), True),
])

# Convert each Document instance into a Row
rows = []
for doc in documents:
    row = (
        doc.id_,
        doc.metadata["EventID"],
        doc.metadata["TyphoonName"],
        doc.metadata["ObservationDate"],
        doc.metadata["ObservationTime"],
        doc.metadata["ObservationDateTime"],
        doc.metadata["InfoType"],
        doc.metadata["Serial"],
        doc.metadata["InfoKindVersion"],
        doc.text,
        doc.embedding
    )
    rows.append(row)

# Create a DataFrame from the list of rows
df = spark.createDataFrame(rows, schema)

# COMMAND ----------

import pyspark.sql.functions as F
df = df.withColumn("unique_event_id", F.concat(F.col("EventID"), F.col("ObservationDateTime")))

# COMMAND ----------

display(df, limit=5)

# COMMAND ----------

# Check if the table exists

table_exists = spark.sql(f"SHOW TABLES IN {catalog}.{Schema} LIKE '{table_name}'").count() > 0

if not table_exists:
    # Create the table if it doesn't exist
    df.write.format("delta").saveAsTable(f"{catalog}.{Schema}.{table_name}")
else:
    # Create a temporary view for the DataFrame
    df.createOrReplaceTempView("df_temp_view")
    
    # Perform the upsert if the table exists
    spark.sql(f"""
        MERGE INTO {catalog}.{Schema}.{table_name} AS target
        USING df_temp_view AS source
        ON target.unique_event_id = source.unique_event_id
        WHEN MATCHED THEN
            UPDATE SET *
        WHEN NOT MATCHED THEN
            INSERT *
    """)

# COMMAND ----------

display(df)

# COMMAND ----------

# # Enable Change Data Feed for the Delta table - only needed once

source_table_fullname = f"{catalog}.{Schema}.{table_name}"
spark.sql(f"""
ALTER TABLE {source_table_fullname}
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # 06. Self Managed Vector Search Index

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Create a Vector Search Endpoint

# COMMAND ----------

vsc = VectorSearchClient(disable_notice=True)

# Verify if the vector search endpoint already exists, if not, create it
try:
    # Check if the vector search endpoint exists
    vsc.get_endpoint(name=vs_endpoint_name)
    print("Endpoint found: " + vs_endpoint_name)
except Exception as e:
    print("\nEndpoint not found: " + vs_endpoint_name)
    # Create a new vector search endpoint
    if "NOT_FOUND" in str(e):
        print("\nCreating Endpoint...")
        vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")
    print("Endpoint Created: " + vs_endpoint_name)

# COMMAND ----------

# check the status of the endpoint
wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)
print(f"Endpoint named {vs_endpoint_name} is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Create Managed Vector Search Index

# COMMAND ----------

# The Delta table containing the content
source_table_fullname = "workspace_us_east_2.default.jma_typhoon_data_03"

# The Delta table to store vector search index
vs_index_fullname = f"{catalog}.{Schema}.jma_typhoon_data_managed_vs_index3"

# COMMAND ----------

def index_exists(vsc, endpoint_name, index_full_name):
  try:
      dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
      return dict_vsindex.get('status').get('ready', False)
  except Exception as e:
      if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
          print(f'Unexpected error describing the index. This could be a permission issue.')
          raise e
  return False

# COMMAND ----------

# create or sync the index from the source column
if not index_exists(vsc, vs_endpoint_name, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {vs_endpoint_name}...")
  
  vsc.create_delta_sync_index(
    endpoint_name=vs_endpoint_name,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="unique_event_id",
    embedding_source_column="Text",
    embedding_model_endpoint_name="databricks-bge-large-en"
  )
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  vsc.get_index(vs_endpoint_name, vs_index_fullname).sync()

# COMMAND ----------

#Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname)
print(f"Index {vs_index_fullname} is ready")
