# Databricks notebook source
!pip install xmltodict
!pip install llama-index-core~=0.10.59 llama-index-embeddings-text-embeddings-inference~=0.1.4 llama-index-llms-openai-like~=0.1.3 msal~=1.28.0 openai~=1.30.1 msal~=1.28.0 "unstructured[pdf,docx]" transformers mlflow databricks-vectorsearch==0.22 flashrank==0.2.0 langchain==0.1.16 databricks-sdk

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
import time
import re
import sys
from datetime import datetime
from pathlib import Path
from datetime import datetime

sys.path.append("../utils")
import nest_asyncio

nest_asyncio.apply()

# Load parameters from YAML file
yaml_file = open("utils/parameters.yaml", "r")
variables = yaml.safe_load(yaml_file)

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

# COMMAND ----------

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Document,
)

# from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.core.schema import MetadataMode

# COMMAND ----------

# MAGIC %md
# MAGIC # 01. Parameters

# COMMAND ----------

catalog = variables['CATALOG']
Schema = variables['SCHEMA']
volume = variables['VOLUME']
directory = variables['DIRECTORY']
table_name = variables['TABLE_NAME']
index_name = variables['INDEX_NAME']
vs_endpoint_name = variables['VS_ENDPOINT_NAME']
model_name = variables['MODEL_NAME']

# COMMAND ----------

# MAGIC %md
# MAGIC # 02. Self Managed Vector Search Index

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Wait for Vector Search Endpoint

# COMMAND ----------

vsc = VectorSearchClient(disable_notice=True)

# Verify if the vector search endpoint already exists
try:
    # Check if the vector search endpoint exists
    vsc.get_endpoint(name=vs_endpoint_name)
    print("Endpoint found: " + vs_endpoint_name)
except Exception as e:
    print("\nEndpoint not found: " + vs_endpoint_name)

# COMMAND ----------

# check the status of the endpoint
wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name)
print(f"Endpoint named {vs_endpoint_name} is ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Wait for Vector Search Index

# COMMAND ----------

# The Delta table containing the text embeddings and metadata
source_table_fullname = f"{catalog}.{Schema}.{table_name}"

# The Delta table to store vector search index
vs_index_fullname = f"{catalog}.{Schema}.{index_name}"

# COMMAND ----------

# Check if index is in ready condition
wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_fullname)
print(f"Index {vs_index_fullname} is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC # 03. Search Document Similar to Query

# COMMAND ----------

# get vs index
index = vsc.get_index(vs_endpoint_name, vs_index_fullname)

question = "What is the position of Typhoon Trami?"

# search
results = index.similarity_search(
    query_text=question, columns=["id_", "Text"], num_results=4
)

# show the results
docs = results.get("result", {}).get("data_array", [])

pp.pprint(docs)

# COMMAND ----------

# MAGIC %md
# MAGIC # 04. Assembling RAG

# COMMAND ----------

# MAGIC %md
# MAGIC ## A. Set up Retriever

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings

# COMMAND ----------

# Define embedding model
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# COMMAND ----------

# test retriever
vectorstore = get_retriever()
similar_documents = vectorstore.invoke(
    "What is position of Typhoon Trami on 2024-08-30T12:00:00?"
)
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## B. Setup Foundation Model

# COMMAND ----------

# Import necessary libraries
from langchain.chat_models import ChatDatabricks

# Define foundation model for generating responses
chat_model = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct", max_tokens=300
)

# Test foundation model
print(f"Test chat model: {chat_model.invoke('What is Generative AI?')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## C. Retrieval

# COMMAND ----------

# MAGIC %md
# MAGIC Set up Prompt Template

# COMMAND ----------

from langchain import LLMChain, PromptTemplate

date_parse_prompt_template = PromptTemplate(
    template="""
        Extract date and time from the following text: {text}
        Just return the typhoon name, date and correponding time if any, no reasoning needed
        For example, for this question 'what is the Shanshan position on 28 Oct 2024 1pm?' return 'Shanshan, 2024-08-28T13:00:00, None'.

        If there are more than one date and time set, return them accordingly.
        For each of the date time set, if there is no date or time in the text, return None
        If there is date but no time in the text, return only date in 'YYYY-MM-DD'
        Otherwise, return the date times in ISO format 'YYYY-MM-DDTHH:mm:ss'
        For example, for this question 'What is the trajectory of Typhoon Shanshan from 27 Aug 2024 to 28 Aug 2024 and then 29 Aug 2024 8pm?', you will return 'Shanshan, 2024-08-27, 2024-08-28, 2024-08-29T20:00:00'
        When asked for information up to or until certain date, return the event name, None start_date and the date as end_date. 
        For example, for this question 'What is the trajectory of Typhoon Shanshan up to 28 Aug 2024?', you will return 'Shanshan, None, 2024-08-28',
        When asked for information up to or until 'end' of certain date, return the event name, None, and the end date at time 23:59:00.
        For example, for this question 'What is the trajectory of Typhoon Shanshan up to end of 29 Aug 2024?', you will return 'Shanshan, None, 2024-08-29T23:59:00', 
        For example, for this question 'What is the trajectory of Typhoon Kong Rey until end of 30 Aug 2024?', you will return 'Kong Rey, None, 2024-08-30T23:59:00'
    """,
    input_variables=["text"],  # Variable for user-provided text
)

# COMMAND ----------

# MAGIC %md
# MAGIC Assemble prompt by adding query

# COMMAND ----------

# query = "tell me about yourself"
# query="what is the Shanshan position on 29 Aug 2024"
# query = "What is the trajectory of Typhoon Trami up to end of 29 Oct 2024?"
# query = (
#     "What is the trajectory of Typhoon Shanshan from 27 Aug 2024 1pm to 29 Aug 2024 9pm?"
# )
# query = "What is the trajectory of Kong Rey from 28 Oct 2024 6 am to 29 Oct 2024 10pm?"
# query = "What is shanshan?"
# query = "what is Shanshan position at the end of 29 Aug 2024 at 11pm?"
# query = "Where is Shanshan on 29 Aug 2024 at 23:00:00?"
query = "What is Shanshan trajectory from 28 Aug 2024 to 29 Aug 2024?"

# replace text with query
date_parse_prompt = date_parse_prompt_template.format(
    text=query
)

# COMMAND ----------

# MAGIC %md
# MAGIC Call LLM to Parse Date and Typhoon Name from query

# COMMAND ----------

parsed_content = chat_model.invoke(date_parse_prompt).content
parsed_content

# COMMAND ----------

check_format(parsed_content)

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import col, to_timestamp,from_utc_timestamp, to_date, lit

# Check if the parsed_content is suitable to retrieve specific documents
if check_format(parsed_content) == False:
    docs = vectorstore.invoke(query)
    try:
        rag_context = " ".join(doc.page_content for row in docs)
    except:
        rag_context = " ".join([doc.page_content for doc in docs])
else:
    dates = parsed_content.split(", ")
    print(dates)

    # Format parsed query
    typhoon_name_upper = dates[0].upper().replace("TYPHOON", "").strip()
    print(f"Parsed typhoon name: {typhoon_name_upper}")

    # Extract text
    # Read table, convert ObservationDateTime from string to timestamp, take note of the time zone
    df = spark.read.format("delta").table(f"{catalog}.{Schema}.{table_name}")
    df = df.withColumn("ObservationDateTime_Timestamp", to_timestamp(col("ObservationDateTime"), "yyyy-MM-dd'T'HH:mm:ssXXX"))
    df = df.withColumn("ObservationDateTime", from_utc_timestamp(col("ObservationDateTime_Timestamp"), "Asia/Tokyo"))
    # Format typhoon name, remove hyphen for ease of matching
    df = df.withColumn(
        "TyphoonName_no_hyphen", F.regexp_replace(F.col("TyphoonName"), "-", "")
    )

    # If parsed input has 1 or more components
    if len(dates) >= 1:
        try: 
            start = dates[1]
            start_date = parse_date_time(start)
            print(f"Start date: {start_date}")
        except:
            start_date = None
        try:
            end = dates[2]
            end_date = parse_date_time(end)
            print(f"End date: {end_date}")
        except:
            end_date = None
    else:
        print("Non Event Specific Query")
        start = None
        end = None
    
    # Prepare filtering condition
    if start_date and end_date:
        filter_condition = (
            to_timestamp(col("ObservationDateTime")).between(start_date, end_date)
            & (
                col("TyphoonName").like(f"%{typhoon_name_upper}%")
                | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%")
                | col("TyphoonName_no_hyphen").rlike(f"{typhoon_name_upper}[A-Z]+")
                | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%".replace(" ", ""))
            )
        )
    elif start_date:
        # if contain only date, without time
        if start_date.time() == datetime.min.time():

            filter_condition = (
                (to_date(col("ObservationDateTime")) == to_date(lit(start_date)))
                & (
                    col("TyphoonName").like(f"%{typhoon_name_upper}%")
                    | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%")
                    | col("TyphoonName_no_hyphen").rlike(f"{typhoon_name_upper}[A-Z]+")
                    | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%".replace(" ", ""))
                )
            )
        # if date and time
        else:
            filter_condition = (
                (to_timestamp(col("ObservationDateTime")) == to_timestamp(lit(start_date.isoformat())))
                & (
                    col("TyphoonName").like(f"%{typhoon_name_upper}%")
                    | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%")
                    | col("TyphoonName_no_hyphen").rlike(f"{typhoon_name_upper}[A-Z]+")
                    | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%".replace(" ", ""))
                )
        )
    elif end_date:
        filter_condition = (
            (col("ObservationDateTime") <= end_date)
            & (
                col("TyphoonName").like(f"%{typhoon_name_upper}%")
                | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%")
                | col("TyphoonName_no_hyphen").rlike(f"{typhoon_name_upper}[A-Z]+")
                | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%".replace(" ", ""))
            )
        )
    else:
        filter_condition = (
            col("TyphoonName").like(f"%{typhoon_name_upper}%")
            | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%")
            | col("TyphoonName_no_hyphen").rlike(f"{typhoon_name_upper}[A-Z]+")
            | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%".replace(" ", ""))
        )

    # Collect filtered row from documents table
    filtered_df = df.filter(filter_condition).select("ObservationDate", "ObservationTime","ObservationDateTime","Text", "TyphoonName")

    display(filtered_df)

    # Collect text content
    text_context = filtered_df.select("Text").collect()

    # Prepare retrieved context for next step
    rag_context = " ".join(row.Text for row in text_context)

# COMMAND ----------

rag_context

# COMMAND ----------

# MAGIC %md
# MAGIC ## D. Output Generation

# COMMAND ----------

#  Import necessary libraries
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# COMMAND ----------

answer_prompt_template = PromptTemplate(
    template="""
        You are a helpful AI bot with expertise in typhoon data.
        If you are asked about matters out of this scope, kindly reply your inability to answer.
        Based on the context {context}, answer the {query}.
        Provide factual and concise answer.
        When asked about trajectory, use the context and conclude the movement from the beginning of the period to the end of the period.
        When asked about position at specific date and time, answer with the precise Latitude, Longitude and location narrative.
    """,
    input_variables=["context", "query"],  # Variable for user-provided text
)

answer_prompt = answer_prompt_template.format(context=rag_context, query=query)

# COMMAND ----------

answer = chat_model.invoke(answer_prompt).content
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ##E. Langchain

# COMMAND ----------

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# COMMAND ----------

# Set up the Conversational Chain
chain = LLMChain(
    llm=chat_model,
    prompt=answer_prompt_template,
    verbose=True
)

# COMMAND ----------

# Run the chain with the manually retrieved context and question
result = chain({"context": rag_context, "query": query})
print(result['text'])

# COMMAND ----------

# MAGIC %md
# MAGIC # 05. Putting all Together

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parse Query and Collect Documents

# COMMAND ----------

# Combine codesfrom langchain import LLMChain, PromptTemplate
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.functions import to_timestamp, to_date, lit

# Step 1: Define the LLM date parsing function
def parse_date_from_query(query):
    date_parse_prompt_template = PromptTemplate(
        template="""
            Extract date and time from the following text: {text}
            Just return the typhoon name, date and corresponding time if any, no reasoning needed
            If there are more than one date and time set, return them accordingly.
            For each of the date time set, if there is no date or time in the text, return None
            If there is date but no time in the text, return only date in 'YYYY-MM-DD'
            Otherwise, return the date times in ISO format 'YYYY-MM-DDTHH:mm:ss'
            For example, for this question 'What is the trajectory of Typhoon Shanshan from 27 Aug 2024 to 28 Aug 2024 and then 29 Aug 2024 8pm?', you will return 'Shanshan, 2024-08-27, 2024-08-28, 2024-08-29T20:00:00'
            When asked for information up to or until certain date, return the event name, None start_date and the date as end_date. 
            For example, for this question 'What is the trajectory of Typhoon Shanshan up to 28 Aug 2024?', you will return 'Shanshan, None, 2024-08-28',
            When asked for information up to or until 'end' of certain date, return the event name, None, and the end date at time 23:59:00.
            For example, for this question 'What is the trajectory of Typhoon Shanshan up to end of 29 Aug 2024?', you will return 'Shanshan, None, 2024-08-29T23:59:00', 
            For example, for this question 'What is the trajectory of Typhoon Kong Rey until end of 30 Aug 2024?', you will return 'Kong Rey, None, 2024-08-30T23:59:00'
        """,
        input_variables=["text"],
    )

    date_parse_prompt = date_parse_prompt_template.format(text=query)
    
    # Assume chat_model is already defined and initialized with an appropriate LLM
    parsed_content = chat_model.invoke(date_parse_prompt).content
    return parsed_content

# Step 2: Define the document collection function
def collect_documents_based_on_parsed_content(parsed_content, spark, catalog, schema):
    from datetime import datetime

    # Check if the parsed_content is suitable to retrieve specific documents
    if check_format(parsed_content) == False:
        docs = vectorstore.invoke(query)
        try:
            rag_context = " ".join(doc.page_content for row in docs)
        except:
            rag_context = " ".join([doc.page_content for doc in docs])
    else:
        dates = parsed_content.split(", ")
        print(dates)

        # Extract text
        df = spark.read.format("delta").table(f"{catalog}.{Schema}.{table_name}")
        df = df.withColumn("ObservationDateTime_Timestamp", to_timestamp(col("ObservationDateTime"), "yyyy-MM-dd'T'HH:mm:ssXXX"))
        df = df.withColumn("ObservationDateTime", from_utc_timestamp(col("ObservationDateTime_Timestamp"), "Asia/Tokyo"))
        df = df.withColumn(
            "TyphoonName_no_hyphen", F.regexp_replace(F.col("TyphoonName"), "-", "")
        )

        typhoon_name_upper = dates[0].upper().replace("TYPHOON", "").strip()
        print(f"Parsed typhoon name: {typhoon_name_upper}")
        
        print(len(dates))

        if len(dates) >= 2:
            start = dates[1]
            start_date = parse_date_time(start)
            print(f"Start date: {start_date}")
            try:
                end = dates[2]
                end_date = parse_date_time(end)
                print(f"End date: {end_date}")
            except:
                end_date = None
        else:
            print("Non Event Specific Query")
            start = None
            end = None
        
        # Prepare filtering condition
        if start_date and end_date:
            filter_condition = (
                to_timestamp(col("ObservationDateTime")).between(start_date, end_date)
                & (
                    col("TyphoonName").like(f"%{typhoon_name_upper}%")
                    | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%")
                    | col("TyphoonName_no_hyphen").rlike(f"{typhoon_name_upper}[A-Z]+")
                    | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%".replace(" ", ""))
                )
            )
        elif start_date:
            # if contain only date, without time
            if start_date.time() == datetime.min.time():

                filter_condition = (
                    (to_date(col("ObservationDateTime")) == to_date(lit(start_date)))
                    & (
                        col("TyphoonName").like(f"%{typhoon_name_upper}%")
                        | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%")
                        | col("TyphoonName_no_hyphen").rlike(f"{typhoon_name_upper}[A-Z]+")
                        | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%".replace(" ", ""))
                    )
                )
            # if date and time
            else:
                filter_condition = (
                    (to_timestamp(col("ObservationDateTime")) == to_timestamp(lit(start_date.isoformat())))
                    & (
                        col("TyphoonName").like(f"%{typhoon_name_upper}%")
                        | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%")
                        | col("TyphoonName_no_hyphen").rlike(f"{typhoon_name_upper}[A-Z]+")
                        | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%".replace(" ", ""))
                    )
            )
        elif end_date:
            filter_condition = (
                (col("ObservationDateTime") <= end_date)
                & (
                    col("TyphoonName").like(f"%{typhoon_name_upper}%")
                    | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%")
                    | col("TyphoonName_no_hyphen").rlike(f"{typhoon_name_upper}[A-Z]+")
                    | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%".replace(" ", ""))
                )
            )
        else:
            filter_condition = (
                col("TyphoonName").like(f"%{typhoon_name_upper}%")
                | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%")
                | col("TyphoonName_no_hyphen").rlike(f"{typhoon_name_upper}[A-Z]+")
                | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%".replace(" ", ""))
            )
  
    filtered_df = df.filter(filter_condition).select("ObservationDateTime", "Text", "TyphoonName")
    text_context = filtered_df.select("Text").collect()
    
    # Prepare retrieved context for next step
    rag_context = " ".join(row.Text for row in text_context)
    
    return rag_context


# COMMAND ----------

# MAGIC %md
# MAGIC ### Main Chain

# COMMAND ----------

from langchain import LLMChain, PromptTemplate
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.sql.functions import to_timestamp, to_date, lit

# Step 3: Define the main chain logic
class CustomChain:
    def __init__(self, llm, spark, catalog, schema):
        self.llm = llm
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.prompt_template = answer_prompt_template

    def run(self, user_query):
        # Use LLM to parse date and typhoon name from user query
        parsed_content = parse_date_from_query(user_query)

        # Collect documents based on parsed content
        rag_context = collect_documents_based_on_parsed_content(parsed_content, self.spark, self.catalog, self.schema)

        # Create an LLM chain
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=True)

        # Run the chain and return the response
        return chain.run({"context": rag_context, "query":user_query})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Custom Chain

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
llm = chat_model 
catalog = catalog
schema = Schema

# COMMAND ----------

# Create the main chain instance
custom_chain = CustomChain(llm, spark, catalog, schema)

# COMMAND ----------

# call the custom chain
# query = "Can you show me the trajectory of Typhoon Shanshan until end of 29 Aug 2024?"
# query = "what is Shanshan position at the end of 29 Aug 2024 at 11pm?"
response = custom_chain.run(query)
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC #06. Save Model to Model Registry

# COMMAND ----------

# Import necessary libraries
from mlflow.models import infer_signature
import mlflow.pyfunc
import mlflow
import langchain

# COMMAND ----------

class CustomChainWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, custom_chain):
        self.custom_chain = custom_chain

    def predict(self, context, model_input):
        # If the input is a DataFrame with a 'query' column
        # user_query = model_input["query"].iloc[0]  
        user_query = model_input
        return self.custom_chain.run(user_query) 

# COMMAND ----------

# Initialize the LLM, Spark session, catalog, and schema for your CustomChain
# For example:
custom_chain_instance = CustomChain(llm, spark, catalog, schema)  # Ensure all dependencies are initialized

# Wrap CustomChain instance as pyfunc
custom_chain_model = CustomChainWrapper(custom_chain_instance)

# COMMAND ----------

# for models in unity catalog ensure the registry uri is databricks-uc
mlflow.set_registry_uri("databricks-uc") 
# Model names in UC will need catalog and schema followed by model name
model_fullname = f"{catalog}.{Schema}.{model_name}"
model_fullname

# COMMAND ----------

# Get the Python version
python_version = sys.version.split()[0] 

# COMMAND ----------

# Log the model as a PyFunc
with mlflow.start_run() as run:
    # Define an example input for the model
    input_example = "Where is Shanshan on 29 Aug 2024 at 23:00:00?"

    mlflow.pyfunc.log_model(
        artifact_path=model_name,
        python_model=custom_chain_model,
        registered_model_name=model_fullname,
        input_example=input_example,
        conda_env={
            'channels': ['defaults'],
            'dependencies': [
                f'python={python_version}',
                'pip',
                {
                    'pip': [
                        f'mlflow=={mlflow.__version__}',
                        f'langchain=={langchain.__version__}',
                        # f'pyspark=={pyspark.__version__}',
                        "databricks-vectorsearch",
                    ],
                },
            ],
            'name': 'custom_chain_env'
        }
    )

    # Register the model separately using the artifact URI
    artifact_uri = f"runs:/{run.info.run_id}/{model_name}"
    mlflow.register_model(model_uri=artifact_uri, name=model_fullname)

# COMMAND ----------

# MAGIC %md
# MAGIC # 07. Load Saved Model and Test

# COMMAND ----------

logged_model = artifact_uri

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


# COMMAND ----------

loaded_model.predict("What is Shanshan trajectory from 28 Aug 2024 to 29 Aug 2024?")

# COMMAND ----------


