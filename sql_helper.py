import mysql.connector

def insert_into_sql(image_name, milvus_id, host='localhost', user='ayush', password='ayush', database='frugalproject'):
    try:
        #Establishing Connection
        conn = mysql.connector.connect(host=host,user=user,password=password,database=database)
        cursor = conn.cursor()

        #Creating Table
        create_table_query = "CREATE TABLE IF NOT EXISTS image_vectors (image_name VARCHAR(255), milvus_id BIGINT);"
        cursor.execute(create_table_query)
        conn.commit()

        #Inserting Data
        insert_query = "INSERT INTO image_vectors (image_name, milvus_id) VALUES (%s, %s);"
        val = (image_name, milvus_id)
        cursor.execute(insert_query, val)
        conn.commit()
        #print(cursor.rowcount, "record inserted.")
    except Exception as e:
        print("Exception occured:{}".format(e))
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
            #print("MySQL connection is closed")

def retrieve_results(feat_list,host='localhost',user='ayush',password='ayush',database='frugalproject'):
    try:
        conn = mysql.connector.connect(host=host,user=user,password=password,database=database)
        cursor = conn.cursor()
        query = "SELECT image_name, milvus_id FROM image_vectors WHERE milvus_id IN ({})".format(', '.join(map(str, feat_list)))
        cursor.execute(query)
        results = cursor.fetchall()
        filenames = [row[0] for row in results]
        #print(filenames)
        return filenames
    except Exception as e:
        print("Exception occured:{}".format(e))
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
            print("MySQL connection is closed")