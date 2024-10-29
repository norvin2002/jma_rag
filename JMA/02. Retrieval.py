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
# query="what is the typhoon on 28 Oct 2024?"
# query = "What is the trajectory of Typhoon Trami up to end of 29 Oct 2024?"
# query = (
#     "What is the trajectory of Typhoon Shanshan from 27 Aug 2024 1pm to 29 Aug 2024 9pm?"
# )
query = "What is the trajectory of Kong Rey from 28 Oct 2024 6 am to 29 Oct 2024 10pm?"
# query = "What is shanshan?"

date_parse_prompt = date_parse_prompt_template.format(
    text=query
)  # Replace {text} with query

# COMMAND ----------

# MAGIC %md
# MAGIC Call LLM to Parse Date and Typhoon Name from query

# COMMAND ----------

parsed_content = chat_model.invoke(date_parse_prompt).content

# COMMAND ----------

from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.functions import to_timestamp, to_date, lit

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
    df = spark.read.format("delta").table(f"{catalog}.{Schema}.jma_typhoon_data_03")
    df = df.withColumn('ObservationDateTime', to_timestamp('ObservationDateTime'))
    df = df.withColumn(
        "TyphoonName_no_hyphen", F.regexp_replace(F.col("TyphoonName"), "-", "")
    )

    typhoon_name_upper = dates[0].upper().replace("TYPHOON", "").strip()
    print(f"Parsed typhoon name: {typhoon_name_upper}")
        
    if len(dates) >= 3:
        start = dates[1]
        start_date = parse_date_time(start)
        print(f"Start date: {start_date}")
        end = dates[2]
        end_date = parse_date_time(end)
        print(f"End date: {end_date}")
    else:
        print("Non Event Specific Query")
        start = None
        end = None
    
    # Prepare filtering condition
    if start_date and end_date:
        filter_condition = (
            col("ObservationDateTime").between(start_date, end_date)
            & (
                col("TyphoonName").like(f"%{typhoon_name_upper}%")
                | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%")
                | col("TyphoonName_no_hyphen").rlike(f"{typhoon_name_upper}[A-Z]+")
                | col("TyphoonName_no_hyphen").like(f"%{typhoon_name_upper}%".replace(" ", ""))
            )
        )
    elif start_date:
        filter_condition = (
            (to_date(col("ObservationDateTime")) == to_date(lit(start_date)))
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
    # print(filter_condition)

    filtered_df = df.filter(filter_condition).select("ObservationDateTime","Text", "TyphoonName")

    display(filtered_df)

    text_context = filtered_df.select("Text").collect()

    # Prepare retrieved context for next step
    rag_context = " ".join(row.Text for row in text_context)

# COMMAND ----------

# rag_context

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
# MAGIC #05. Save Model to Model Registry

# COMMAND ----------

# # Import necessary libraries
# from mlflow.models import infer_signature
# import mlflow
# import langchain

# COMMAND ----------

# # Set Model Registry URI to UC

# mlflow.set_registry_uri("databricks-uc-jma")
# model_name = f"{catalog}.{Schema}.jma_rag_app_1"

# COMMAND ----------

# # Register the assembled RAG model in Model Registry with Unity Catalog
# with mlflow.start_run(run_name="jma_rag_app_1") as run:
#     signature = infer_signature(query, answer)
#     model_info = mlflow.langchain.log_model(
#         chain,
#         loader_fn=get_retriever,
#         artifact_path="chain",
#         registered_model_name=model_name,
#         pip_requirements=[
#             "mlflow==" + mlflow.__version__,
#             "langchain==" + langchain.__version__,
#             "databricks-vectorsearch",
#         ],
#         input_example=question,
#         signature=signature
#     )
