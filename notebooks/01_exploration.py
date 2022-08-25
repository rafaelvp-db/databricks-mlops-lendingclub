# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Ensuring Consistency with ACID Transactions with Delta Lake (Loan Risk Data)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC This is a companion notebook to provide a Delta Lake example against the Lending Club data.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## The Data
# MAGIC 
# MAGIC The data used is public data from Lending Club. It includes all funded loans from 2012 to 2017. Each loan includes applicant information provided by the applicant as well as the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. For a full view of the data please view the data dictionary available [here](https://resources.lendingclub.com/LCDataDictionary.xlsx).
# MAGIC 
# MAGIC 
# MAGIC ![Loan_Data](https://preview.ibb.co/d3tQ4R/Screen_Shot_2018_02_02_at_11_21_51_PM.png)
# MAGIC 
# MAGIC https://www.kaggle.com/wendykan/lending-club-loan-data

# COMMAND ----------

# MAGIC %md ## Reading our data

# COMMAND ----------

# DBTITLE 0,Import Data and create pre-Databricks Delta Table
# -----------------------------------------------
# Uncomment and run if this folder does not exist
# -----------------------------------------------
# Configure location of loanstats_2012_2017.parquet
lspq_path = "/databricks-datasets/samples/lending_club/parquet/"

# Read loanstats_2012_2017.parquet
data = spark.read.parquet(lspq_path)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Saving our data into the Delta Lake

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE DATABASE IF NOT EXISTS robeco

# COMMAND ----------

# Write the data to its target.
data.write \
  .format("delta") \
  .saveAsTable("robeco.loan")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Data Exploration

# COMMAND ----------

# DBTITLE 1,Quick Glance
# MAGIC %sql
# MAGIC 
# MAGIC select * from robeco.loan limit 100

# COMMAND ----------

# DBTITLE 1,Quickly Generating a Data Profile
# MAGIC %sql
# MAGIC 
# MAGIC select count(*) from robeco.loan

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM robeco.loan TABLESAMPLE (10 PERCENT);

# COMMAND ----------

# DBTITLE 1,How does our credit quality look like?
# MAGIC %sql
# MAGIC 
# MAGIC select loan_status from robeco.loan

# COMMAND ----------

# DBTITLE 1,How do our credit grades look like?
# MAGIC %sql
# MAGIC 
# MAGIC select grade from robeco.loan

# COMMAND ----------

data = data.filter("addr_state not in ('debt_consolidation', '531xx', 'null')")

# COMMAND ----------

# DBTITLE 1,How are our loans distributed geographically?
# View map of our asset data
from pyspark.sql import functions as F

display(data.groupBy("addr_state").agg((F.count(F.col("annual_inc"))).alias("ratio")))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's review our current set of loans using our map visualization.

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Delta Lake Logo Tiny](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Full DML Support
# MAGIC 
# MAGIC **Note**: Full DML Support is a feature that will be coming soon to Delta Lake; the preview is currently available in Databricks.
# MAGIC 
# MAGIC Delta Lake supports standard DML including UPDATE, DELETE and MERGE INTO providing developers more controls to manage their big datasets.

# COMMAND ----------

# MAGIC %md Let's start by creating a traditional Parquet table

# COMMAND ----------

# Load new DataFrame based on current Delta table
lbs_df = sql("select * from loan_by_state_delta")

# Save DataFrame to Parquet
lbs_df.write.mode("overwrite").parquet("/tmp/loan_by_state.parquet")

# Reload Parquet Data
lbs_pq = spark.read.parquet("/tmp/loan_by_state.parquet")

# Create new table on this parquet data
lbs_pq.createOrReplaceTempView("loan_by_state_pq")

# Review data
display(sql("select * from loan_by_state_pq"))

# COMMAND ----------

# MAGIC %md ###![Delta Lake Logo Tiny](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) DELETE Support
# MAGIC 
# MAGIC The data was originally supposed to be assigned to `WA` state, so let's `DELETE` those values assigned to `IA`

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Attempting to run `DELETE` on the Parquet table
# MAGIC -- DELETE FROM loan_by_state_pq WHERE addr_state = 'IA'

# COMMAND ----------

# MAGIC %md **Note**: This command fails because the `DELETE` statements are not supported in Parquet, but are supported in Delta Lake.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Running `DELETE` on the Delta Lake table
# MAGIC DELETE FROM loan_by_state_delta WHERE addr_state = 'IA'

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Review current loans within the `loan_by_state_delta` Delta Lake table
# MAGIC select addr_state, sum(`count`) as loans from loan_by_state_delta group by addr_state

# COMMAND ----------

# MAGIC %md ###![Delta Lake Logo Tiny](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) UPDATE Support
# MAGIC The data was originally supposed to be assigned to `WA` state, so let's `UPDATE` those values

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Attempting to run `UPDATE` on the Parquet table
# MAGIC -- UPDATE loan_by_state_pq SET `count` = 2700 WHERE addr_state = 'WA'

# COMMAND ----------

# MAGIC %md **Note**: This command fails because the `UPDATE` statements are not supported in Parquet, but are supported in Delta Lake.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Running `UPDATE` on the Delta Lake table
# MAGIC UPDATE loan_by_state_delta SET `count` = 2700 WHERE addr_state = 'WA'

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Review current loans within the `loan_by_state_delta` Delta Lake table
# MAGIC select addr_state, sum(`count`) as loans from loan_by_state_delta group by addr_state

# COMMAND ----------

# MAGIC %md Instead of writing separate `INSERT` and `UPDATE` statements, we can use a `MERGE` statement. 

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO loan_by_state_delta as d
# MAGIC USING merge_table as m
# MAGIC on d.addr_state = m.addr_state
# MAGIC WHEN MATCHED THEN 
# MAGIC   UPDATE SET d.count = m.count
# MAGIC WHEN NOT MATCHED 
# MAGIC   THEN INSERT (SELECT * FROM m)

# COMMAND ----------

# MAGIC %md ## Run Our Model
# MAGIC Let's try to predict loan grades!

# COMMAND ----------



# COMMAND ----------


