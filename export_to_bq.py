import pandas as pd
from google.cloud import bigquery

 

df = pd.read_csv("sentiment_feedback_new.csv", encoding='latin1' )
df1 = pd.read_csv("results.csv" )
df2 = pd.read_csv("results_vader.csv")

# Initialize a BigQuery client
client = bigquery.Client()

# Define the table ID (project_id.dataset_id.table_id)
table_id = 'project-deepa-777777.db_ai.sentiment_feedback_org'
table_id1 = 'project-deepa-777777.db_ai.results_tblob'
table_id2 = 'project-deepa-777777.db_ai.results_vader'

# Upload the DataFrame to BigQuery
job = client.load_table_from_dataframe(df, table_id)
# job1 = client.load_table_from_dataframe(df1, table_id1)
# job2 = client.load_table_from_dataframe(df2, table_id2)

# Wait for the job to complete
job.result()
# job1.result()
# job2.result()

# Verify the results
table = client.get_table(table_id)
print(f"Loaded {table.num_rows} rows to {table_id}.")

# table1 = client.get_table(table_id1)
# print(f"Loaded {table1.num_rows} rows to {table_id1}.")

# table2 = client.get_table(table_id2)
# print(f"Loaded {table2.num_rows} rows to {table_id2}.")
