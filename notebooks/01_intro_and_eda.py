# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #Evaluating Risk for Loan Approvals
# MAGIC 
# MAGIC ## Business Value
# MAGIC 
# MAGIC Being able to accurately assess the risk of a loan application can save a lender the cost of holding too many risky assets. Rather than a credit score or credit history which tracks how reliable borrowers are, we will generate a score of how profitable a loan will be compared to other loans in the past. The combination of credit scores, credit history, and profitability score will help increase the bottom line for financial institution.
# MAGIC 
# MAGIC Having a interporable model that a loan officer can use before performing a full underwriting can provide immediate estimate and response for the borrower and a informative view for the lender.
# MAGIC 
# MAGIC <a href="https://ibb.co/cuQYr6"><img src="https://preview.ibb.co/jNxPym/Image.png" alt="Image" border="0"></a>
# MAGIC 
# MAGIC This notebook has been tested with *DBR 5.4 ML Beta, Python 3*

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

# DBTITLE 1,Import Data
# Configure location of loanstats_2012_2017.parquet
lspq_path = "/databricks-datasets/samples/lending_club/parquet/"

# Read loanstats_2012_2017.parquet
data = spark.read.parquet(lspq_path)

# Reduce the amount of data (to run on DBCE)
(loan_stats_ce, loan_stats_rest) = data.randomSplit([0.10, 0.90], seed=123)

# Select only the columns needed
loan_stats_ce = loan_stats_ce.select("loan_status", "int_rate", "revol_util", "issue_d", "earliest_cr_line", "emp_length", "verification_status", "total_pymnt", "loan_amnt", "grade", "annual_inc", "dti", "addr_state", "term", "home_ownership", "purpose", "application_type", "delinq_2yrs", "total_acc")

# Print out number of loans
print(str(loan_stats_ce.count()) + " loans opened by Lending Club...")

# COMMAND ----------

# DBTITLE 1,Filter Data and Fix Schema
from pyspark.sql.functions import regexp_replace, substring, trim, round, count, col

print("------------------------------------------------------------------------------------------------")
print("Create bad loan label, this will include charged off, defaulted, and late repayments on loans...")
loan_stats_ce = loan_stats_ce.filter(loan_stats_ce.loan_status.isin(["Default", "Charged Off", "Fully Paid"]))\
                       .withColumn("bad_loan", (~(loan_stats_ce.loan_status == "Fully Paid")).cast("string"))


print("------------------------------------------------------------------------------------------------")
print("Turning string interest rate and revoling util columns into numeric columns...")
loan_stats_ce = loan_stats_ce.withColumn('int_rate', regexp_replace('int_rate', '%', '').cast('float')) \
                       .withColumn('revol_util', regexp_replace('revol_util', '%', '').cast('float')) \
                       .withColumn('issue_year',  substring(loan_stats_ce.issue_d, 5, 4).cast('double') ) \
                       .withColumn('earliest_year', substring(loan_stats_ce.earliest_cr_line, 5, 4).cast('double'))
loan_stats_ce = loan_stats_ce.withColumn('credit_length_in_years', (loan_stats_ce.issue_year - loan_stats_ce.earliest_year))


print("------------------------------------------------------------------------------------------------")
print("Converting emp_length column into numeric...")
loan_stats_ce = loan_stats_ce.withColumn('emp_length', trim(regexp_replace(loan_stats_ce.emp_length, "([ ]*+[a-zA-Z].*)|(n/a)", "") ))
loan_stats_ce = loan_stats_ce.withColumn('emp_length', trim(regexp_replace(loan_stats_ce.emp_length, "< 1", "0") ))
loan_stats_ce = loan_stats_ce.withColumn('emp_length', trim(regexp_replace(loan_stats_ce.emp_length, "10\\+", "10") ).cast('float'))

# COMMAND ----------

# MAGIC %md ## ![Delta Lake Tiny Logo](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Easily Convert Parquet to Delta Lake format
# MAGIC With Delta Lake, you can easily transform your Parquet data into Delta Lake format. 

# COMMAND ----------

# Configure Path
DELTALAKE_GOLD_PATH = "/ml/loan_stats.delta"

# Remove table if it exists
dbutils.fs.rm(DELTALAKE_GOLD_PATH, recurse=True)

# Save table as Delta Lake
loan_stats_ce.write.format("delta").mode("overwrite").save(DELTALAKE_GOLD_PATH)

# Re-read as Delta Lake
loan_stats = spark.read.format("delta").load(DELTALAKE_GOLD_PATH)

# Review data
display(loan_stats)

# COMMAND ----------

# DBTITLE 1,Assert Allocation
display(loan_stats)

# COMMAND ----------

# DBTITLE 1,Munge Data
print("------------------------------------------------------------------------------------------------")
print("Map multiple levels into one factor level for verification_status...")
loan_stats = loan_stats.withColumn('verification_status', trim(regexp_replace(loan_stats.verification_status, 'Source Verified', 'Verified')))

print("------------------------------------------------------------------------------------------------")
print("Calculate the total amount of money earned or lost per loan...")
loan_stats = loan_stats.withColumn('net', round( loan_stats.total_pymnt - loan_stats.loan_amnt, 2))

# COMMAND ----------

# MAGIC %md
# MAGIC ##![Delta Lake Logo Tiny](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Schema Evolution
# MAGIC With the `mergeSchema` option, you can evolve your Delta Lake table schema

# COMMAND ----------

# Add the mergeSchema option
loan_stats.write.option("mergeSchema","true").format("delta").mode("overwrite").save(DELTALAKE_GOLD_PATH)

# COMMAND ----------

# Original Schema
loan_stats_ce.printSchema()

# COMMAND ----------

# New Schema
loan_stats.printSchema()

# COMMAND ----------

# MAGIC %md ### ![Delta Lake Tiny Logo](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) Review Delta Lake Table History
# MAGIC All the transactions for this table are stored within this table including the initial set of insertions, update, delete, merge, and inserts with schema modification

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS loan_stats")
spark.sql("CREATE TABLE loan_stats USING DELTA LOCATION '" + DELTALAKE_GOLD_PATH + "'")

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY loan_stats

# COMMAND ----------

# DBTITLE 1,Feature Distribution and Correlation
display(loan_stats)

# COMMAND ----------

# DBTITLE 1,Loans Per State
display(loan_stats.groupBy("addr_state").agg((count(col("annual_inc"))).alias("ratio")))

# COMMAND ----------

# DBTITLE 1,Asset Allocation
display(loan_stats)
# display(loan_stats.groupBy("bad_loan", "grade").agg((sum(col("net"))).alias("sum_net")))

# COMMAND ----------

# DBTITLE 1,Display munged columns
display(loan_stats.select("net","verification_status","int_rate", "revol_util", "issue_year", "earliest_year", "bad_loan", "credit_length_in_years", "emp_length"))

# COMMAND ----------

# DBTITLE 1,Set Response and Predictor Variables
print("------------------------------------------------------------------------------------------------")
print("Setting variables to predict bad loans")
myY = "bad_loan"
categoricals = ["term", "home_ownership", "purpose", "addr_state",
                "verification_status","application_type"]
numerics = ["loan_amnt","emp_length", "annual_inc","dti",
            "delinq_2yrs","revol_util","total_acc",
            "credit_length_in_years"]
myX = categoricals + numerics

loan_stats2 = loan_stats.select(myX + [myY, "int_rate", "net", "issue_year"])
train = loan_stats2.filter(loan_stats2.issue_year <= 2015).cache()
valid = loan_stats2.filter(loan_stats2.issue_year > 2015).cache()

print(f'training count: {train.count()}')
print(f'validation count: {valid.count()}')

# COMMAND ----------

# MAGIC %sql
# MAGIC USE default;
# MAGIC DROP TABLE IF EXISTS loanstats_train;
# MAGIC DROP TABLE IF EXISTS loanstats_valid;

# COMMAND ----------

# Save training and validation tables for future use
train.write.saveAsTable("loanstats_train")
valid.write.saveAsTable("loanstats_valid")
