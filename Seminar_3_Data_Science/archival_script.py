import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import os

class ArchivalManager:
    """
    Manages data archival from hot storage to cold storage.
    """
    def __init__(self, db_config):
        self.db_config = db_config
        self.archive_dir = 'sprint_3/archives'
        if not os.path.exists(self.archive_dir):
            os.makedirs(self.archive_dir)
    
    def connect(self):
        return psycopg2.connect(**self.db_config)
    
    def archive_old_data(self, table_name, date_column, retention_months=6):
        """
        Archive data older than retention_months from a given table.
        """
        cutoff_date = datetime.now() - timedelta(days=retention_months * 30)
        
        conn = self.connect()
        query = f"SELECT * FROM {table_name} WHERE {date_column} < %s"
        
        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        
        if len(df) > 0:
            # Save to parquet
            archive_file = f"{self.archive_dir}/{table_name}_{datetime.now().strftime('%Y%m%d')}.parquet"
            df.to_parquet(archive_file, compression='gzip')
            print(f"Archived {len(df)} records from {table_name} to {archive_file}")
            
            # Optional: Delete from hot storage (commented out for safety)
            # delete_query = f"DELETE FROM {table_name} WHERE {date_column} < %s"
            # cur = conn.cursor()
            # cur.execute(delete_query, (cutoff_date,))
            # conn.commit()
            # cur.close()
        else:
            print(f"No records to archive from {table_name}")
        
        conn.close()

if __name__ == "__main__":
    # Example configuration (update with actual DB credentials)
    db_config = {
        'host': 'localhost',
        'database': 'flight_insurance',
        'user': 'postgres',
        'password': 'password'
    }
    
    # This is a placeholder - in production, use actual DB config
    print("Archival script configured. Update db_config before production use.")
