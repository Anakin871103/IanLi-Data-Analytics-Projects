# database_interaction.py

import sqlite3

class DatabaseInteraction:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()

    def create_table(self, create_table_sql):
        try:
            self.cursor.execute(create_table_sql)
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Error creating table: {e}")

    def insert_data(self, insert_sql, data):
        try:
            self.cursor.execute(insert_sql, data)
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Error inserting data: {e}")

    def query_data(self, query_sql):
        try:
            self.cursor.execute(query_sql)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error querying data: {e}")
            return None

    def close_connection(self):
        if self.connection:
            self.connection.close()