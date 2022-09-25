import azure.cosmos.documents as documents
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from azure.cosmos.partition_key import PartitionKey
import datetime
import time
import os


class model():
    """
    Just a simple class to handle comunication with CosmosDB
    for saving and downloading the records about ship data.
    """

    def __init__(self):
        """Establishes the connection with CosmosDB"""

        try:
            self.HOST = os.environ['DB_HOST']
            self.MASTER_KEY = os.environ['DB_KEY']
            self.DATABASE_ID = os.environ['DB_ID'] 
        except KeyError as e:
            try:
                print("Unable to find Environmental variables. Attempting to read from config file")
                import config
                self.HOST = config.settings['host']
                self.MASTER_KEY = config.settings['master_key']
                self.DATABASE_ID = config.settings['database_id']
            except Exception as eee:
                print("Unable to import DB config file")


        self.client = cosmos_client.CosmosClient(self.HOST, {'masterKey': self.MASTER_KEY}, user_agent="CosmosDBPythonQuickstart", user_agent_overwrite=True)
        self.db = self.client.get_database_client(self.DATABASE_ID)
        self.con_suspicious = self.db.get_container_client("Suspicious")
        self.con_ships = self.db.get_container_client("Ships")


    def write_suspicious_data(self, json_data):
        """Adds a timestamp, packs the data together and writes json data into Suspicious container"""
        data={"id": str(time.mktime(datetime.datetime.now().timetuple())),
              "records":json_data}
        self.con_suspicious.create_item(body=data)


    def read_latest_suspicious_data(self):
        items = list(self.con_suspicious.query_items(enable_cross_partition_query=True,
        query="SELECT top 1 * FROM c ORDER by c._ts desc"))

        return items[0]

    def write_ships_data(self, json_data):
        """Adds a timestamp, packs the data together and writes json data into Ships container"""
        data={"id": str(time.mktime(datetime.datetime.now().timetuple())),
                "records":json_data}
        self.con_ships.create_item(body=data)


    def read_latest_ships_data(self):
        items = list(self.con_ships.query_items(enable_cross_partition_query=True,
        query="SELECT top 1 * FROM c ORDER by c._ts desc"))

        return items[0]

