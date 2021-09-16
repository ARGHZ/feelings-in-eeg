# coding=utf-8
from pyparsing import unicode
from pymongo import MongoClient, errors
from data_sets_helper import np


class DataBase:

    def __init__(self, host='localhost', port=3001, database_name='meteor'):
        """

        :param host: str
        :param port: int
        :param database_name: str
        """
        self.host, self.port, self.database_name = host, port, database_name

        self.client = MongoClient(self.host, self.port)
        self.database = self.client[self.database_name]

    def close_client(self):
        """
        Close database connection

        """
        self.client.close()

    def insert_one(self, collection_name, document_series):
        """

        :param collection_name: str
        :param document_series: dict
        :return: InsertOneResult
        """
        collection = self.database[collection_name]

        try:
            record = collection.insert_one(document_series)
        except errors.InvalidDocument:
            # Python 2.7.10 on Windows and Pymongo are not forgiving
            # If you have foreign data types you have to convert them
            n = {}
            for k, v in document_series.items():
                if isinstance(k, unicode):
                    for i in ['utf-8', 'iso-8859-1']:
                        try:
                            k = k.encode(i)
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            continue
                if isinstance(v, np.int64):
                    v = int(v)
                if isinstance(v, unicode):
                    for i in ['utf-8', 'iso-8859-1']:
                        try:
                            v = v.encode(i)
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            continue

                n[k] = v

            record = collection.insert(n)

        return record