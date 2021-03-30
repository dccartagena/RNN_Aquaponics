import pandas as pd
import psycopg2
import psycopg2.extras

DB_HOST = "THE_HOST"
DB_PORT = "THE_PORT"
DB_NAME = "DB_NAME"
DB_USER = "YOUR_USER"
DB_PASS = "YOUR_PASSWORD"

# Connection with database
try:
    conn = psycopg2.connect(dbname = DB_NAME,
                            user = DB_USER,
                            password = DB_PASS,
                            host = DB_HOST)
    if conn is not None:
        print('Connected with {} database'.format(DB_NAME))
    else:
        print('Connection not established with {} database'.format(DB_NAME))

except (Exception, psycopg2.DatabaseError) as error:
    print(error)

# Retrieve data in pickle format
cur = conn.cursor()
sql_command = '''SELECT
                    M.id_sensors as "label",
                    M.v_timestamps AS "time",
                    M.v_measurements AS "measurements"

                FROM measurements as M, sensors as S
                WHERE
                    S.id = M.id_sensors
                    AND M.v_timestamps BETWEEN '2021-03-23T00:00:00Z' AND '2021-03-29T23:59:59.00Z'
                    ORDER BY M.id_sensors, M.v_timestamps'''
cur.execute(sql_command)
raw_data = cur.fetchall()

# Close connection
conn.close()

# Create Panda dataframe with raw data
dataset = pd.DataFrame(data = raw_data, columns=['Label', 'DateTime', 'Value'])
dataset['Time'], dataset['Date'] = dataset['DateTime'].apply(lambda x:x.time()), dataset['DateTime'].apply(lambda x:x.date())
dataset.drop(['DateTime'], axis = 1, inplace=True)
dataset.to_csv('dataset_aquaponics_03232021_03292021.txt', sep='\t')
print(dataset)
print("Dataset downloaded")