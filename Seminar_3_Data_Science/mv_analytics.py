import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class MaterializedViewAnalytics:
    """
    Uses the materialized view (mv_monthly_payouts) for analytics and model input.
    """
    def __init__(self, db_config=None):
        self.db_config = db_config
        self.data = None
    
    def load_from_db(self):
        """
        Load data from PostgreSQL materialized view.
        """
        if self.db_config is None:
            print("No DB config provided. Using CSV fallback.")
            return None
        
        conn = psycopg2.connect(**self.db_config)
        query = "SELECT * FROM mv_monthly_payouts ORDER BY month, destination"
        self.data = pd.read_sql_query(query, conn)
        conn.close()
        print(f"Loaded {len(self.data)} records from mv_monthly_payouts")
        return self.data
    
    def load_from_csv(self, csv_path='sprint_3/mv_monthly_payouts.csv'):
        """
        Load data from CSV (for offline analysis).
        """
        if not os.path.exists(csv_path):
            # Create sample data for demonstration
            self.data = pd.DataFrame({
                'destination': ['NYC', 'LAX', 'ORD', 'NYC', 'LAX', 'ORD'] * 6,
                'month': pd.date_range('2023-01-01', periods=36, freq='M'),
                'total_payout': [1000, 1500, 1200, 1100, 1600, 1300] * 6
            })
            self.data.to_csv(csv_path, index=False)
            print(f"Created sample data at {csv_path}")
        else:
            self.data = pd.read_csv(csv_path)
            self.data['month'] = pd.to_datetime(self.data['month'])
        return self.data
    
    def generate_report(self):
        """
        Generate analytics report from materialized view.
        """
        if self.data is None:
            print("No data loaded. Call load_from_db() or load_from_csv() first.")
            return
        
        print("\n=== Materialized View Analytics ===")
        print(f"Total Destinations: {self.data['destination'].nunique()}")
        print(f"Date Range: {self.data['month'].min()} to {self.data['month'].max()}")
        print(f"Total Payout: ${self.data['total_payout'].sum():,.2f}")
        
        print("\nTop 5 Destinations by Total Payout:")
        top_dest = self.data.groupby('destination')['total_payout'].sum().sort_values(ascending=False).head()
        print(top_dest)
        
        print("\nMonthly Trend (Last 6 Months):")
        recent = self.data[self.data['month'] >= self.data['month'].max() - pd.DateOffset(months=6)]
        monthly_trend = recent.groupby('month')['total_payout'].sum()
        print(monthly_trend)
    
    def plot_trends(self, output_path='sprint_3/mv_payout_trends.png'):
        """
        Visualize payout trends.
        """
        if self.data is None:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Monthly total payouts
        monthly = self.data.groupby('month')['total_payout'].sum()
        axes[0].plot(monthly.index, monthly.values, marker='o')
        axes[0].set_title('Total Monthly Payouts')
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Total Payout ($)')
        axes[0].grid(True)
        
        # Top destinations
        top_5_dest = self.data.groupby('destination')['total_payout'].sum().sort_values(ascending=False).head()
        axes[1].bar(top_5_dest.index, top_5_dest.values)
        axes[1].set_title('Top 5 Destinations by Total Payout')
        axes[1].set_xlabel('Destination')
        axes[1].set_ylabel('Total Payout ($)')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
        plt.close()

if __name__ == "__main__":
    import os
    
    # Use CSV fallback for demonstration
    analytics = MaterializedViewAnalytics()
    analytics.load_from_csv()
    analytics.generate_report()
    analytics.plot_trends()
    
    print("\nMaterialized view analytics complete.")
    print("To use with PostgreSQL, provide db_config and call load_from_db()")
