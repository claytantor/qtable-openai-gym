import psycopg2
from psycopg2 import Error

try:
    # Connect to an existing database
    connection = psycopg2.connect(user="postgres",
                                    password="coinbot",
                                    host="127.0.0.1",
                                    port="5432",
                                    database="coinbot")

    # Create a cursor to perform database operations
    cursor = connection.cursor()
    # Print PostgreSQL details
    # print("PostgreSQL server information")

    sql_s = """SELECT id, datetime, open, high, low, close, volume, frequency, security_id 
        FROM security_price 
        WHERE frequency::text='daily' 
            AND security_id=2 ORDER by datetime"""

    cursor.execute(sql_s)
    # print("The number of prices: ", cursor.rowcount)
    row = cursor.fetchone()

    print("Date,Open,High,Low,Close,Adj Close,Volume")
    while row is not None:
        d_f = row[1].strftime("%Y-%m-%d")
        #(7975, datetime.datetime(2021, 8, 31, 0, 0), 46996.8, 47172.04, 46708.93, 47060.06, 1890.99781925, 'daily', 2)
        # Date,Open,High,Low,Close,Adj Close,Volume
        # 2009-05-22,198.528534,199.524521,196.196198,196.946945,196.946945,3433700
        print(f"{d_f},{row[2]},{row[3]},{row[4]},{row[5]},{row[5]},{row[6]}")
        row = cursor.fetchone()

    cursor.close()



except (Exception, Error) as error:
    print("Error while connecting to PostgreSQL", error)
finally:
    if (connection):
        cursor.close()
        connection.close()
        # print("PostgreSQL connection is closed")