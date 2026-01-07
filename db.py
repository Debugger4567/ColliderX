import psycopg2
import os

def get_conn():
    return psycopg2.connect(
        dbname=os.getenv("PGDATABASE", "colliderx"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "Soccer@21"),
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", 5432),
    )