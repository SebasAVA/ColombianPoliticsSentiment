import psycopg2
from sshtunnel import SSHTunnelForwarder
import csv
import os

if not os.path.isdir("data"):
    print("create folder for data")
    os.mkdir("data")

def listToString(l):
    s = ""
    for i in l:
        s += (i + ",")
    s = s[:-1]
    return s

def castData(curs):
    data = []
    for row in curs:
        data.append(row)
    print("Data casted")
    return data

def toCSV(filename,columns,content):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(content)
    print("CSV output file ready")


try:
    with SSHTunnelForwarder(
            ('161.35.123.231', 22),
            ssh_username="postgres",
            ssh_password="dbConn2021!",
            remote_bind_address=('localhost', 5432)) as server:

        print("server connected")

        keepalive_kwargs = {
            "keepalives": 1,
            "keepalives_idle": 60,
            "keepalives_interval": 10,
            "keepalives_count": 5
        }

        params = {
            'database': 'tweetproject',
            'user': 'postgres',
            'password': 'padova2021',
            'host': server.local_bind_host,
            'port': server.local_bind_port,
            **keepalive_kwargs
        }

        conn = psycopg2.connect(**params)
        curs = conn.cursor()
        print("database connected")

        curs.execute("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'tweet'")
        print("\nDownload politicians tweets")
        columns_name = ['twitter_id','id','label']
        curs.execute("SELECT twitter_id, id, full_text FROM tweet WHERE in_reply_twitter_id IS NULL")
        content = castData(curs)
        toCSV('data/politicians_tweets.csv', columns_name,content)

        # print("\nDownload reply tweets")
        # curs.execute("SELECT twitter_id,full_text FROM tweet WHERE in_reply_twitter_id IS NOT NULL")
        # content = castData(curs)
        # toCSV('data/reply_tweets.csv',columns_name,content)
        #
        # print("\nDownload all tweets")
        # columns_name = columns_name = ['twitter_id','id','label','in_reply_twitter_id']
        # curs.execute("SELECT twitter_id,id,full_text,in_reply_twitter_id FROM tweet")
        # content = castData(curs)
        # toCSV('data/tweets.csv', columns_name, content)
        # #
        # print("\nDownload edges")
        # columns_name = ['source','target']
        # curs.execute("SELECT twitter_id,in_reply_twitter_id FROM tweet WHERE in_reply_twitter_id IS NOT NULL")
        # content = castData(curs)
        # toCSV('data/edges.csv',columns_name,content)

        print("\nDownload topics")
        columns_name = ['id_tweet','id_topic','score','is_reply']
        curs.execute("SELECT * FROM tweet_topic")
        content = castData(curs)
        toCSV('data/tweet_topic.csv',columns_name,content)

        print("\nDownload topic dictionary")
        columns_name = ['id_topic','label']
        curs.execute("SELECT * FROM topic")
        content = castData(curs)
        toCSV('data/topic_dictionary.csv',columns_name,content)

except (Exception) as error:
    print("Connection Failed")
    print(error)

