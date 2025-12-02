import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class BusinessKPICalculator:
    """
    Calculate business-level KPIs (not model performance metrics).
    These KPIs measure actual business outcomes and system performance.
    """
    
    def __init__(self, data_dir='sprint_3/data_generation'):
        self.data_dir = data_dir
        self.kpis = {}
        
    def load_data(self):
        """Load all relevant CSV files."""
        print("Loading business data...")
        
        self.users = pd.read_csv(f'{self.data_dir}/User.csv')
        self.bookings = pd.read_csv(f'{self.data_dir}/Booking.csv')
        self.payments = pd.read_csv(f'{self.data_dir}/Payment.csv')
        self.policies = pd.read_csv(f'{self.data_dir}/InsurancePolicy.csv')
        self.claims = pd.read_csv(f'{self.data_dir}/InsuranceClaim.csv')
        self.flights = pd.read_csv(f'{self.data_dir}/Flight.csv')
        
        print(f"  Loaded {len(self.users)} users")
        print(f"  Loaded {len(self.bookings)} bookings")
        print(f"  Loaded {len(self.payments)} payments")
        print(f"  Loaded {len(self.policies)} policies")
        print(f"  Loaded {len(self.claims)} claims")
        
    def calculate_conversion_rate(self):
        """
        Primary KPI: Conversion Rate
        % of users who complete a transaction (have a payment)
        """
        # Merge bookings with payments to get user IDs
        bookings_with_payments = self.bookings.merge(
            self.payments[['bookingId']], 
            left_on='bookingId',
            right_on='bookingId',
            how='inner'
        )
        users_with_payments = bookings_with_payments['user_id'].nunique()
        total_users = len(self.users)
        
        conversion_rate = (users_with_payments / total_users) * 100
        self.kpis['conversion_rate_percent'] = round(conversion_rate, 2)
        
        print(f"\n✓ Conversion Rate: {conversion_rate:.2f}%")
        print(f"  ({users_with_payments}/{total_users} users completed transactions)")
        
    def calculate_policy_purchase_rate(self):
        """
        KPI: Policy Purchase Rate
        % of bookings that resulted in insurance policy purchase
        """
        bookings_with_policies = self.policies['bookingId'].nunique()
        total_bookings = len(self.bookings)
        
        purchase_rate = (bookings_with_policies / total_bookings) * 100
        self.kpis['policy_purchase_rate_percent'] = round(purchase_rate, 2)
        
        print(f"\n✓ Policy Purchase Rate: {purchase_rate:.2f}%")
        print(f"  ({bookings_with_policies}/{total_bookings} bookings purchased insurance)")
        
    def calculate_claim_rate(self):
        """
        KPI: Claim Rate
        % of policies that resulted in a claim
        """
        policies_with_claims = self.claims['policyId'].nunique()
        total_policies = len(self.policies)
        
        claim_rate = (policies_with_claims / total_policies) * 100 if total_policies > 0 else 0
        self.kpis['claim_rate_percent'] = round(claim_rate, 2)
        
        print(f"\n✓ Claim Rate: {claim_rate:.2f}%")
        print(f"  ({policies_with_claims}/{total_policies} policies filed claims)")
        
    def calculate_average_transaction_value(self):
        """
        KPI: Average Transaction Value (ATV)
        Mean payment amount across all transactions
        """
        avg_payment = self.payments['amount'].mean()
        self.kpis['average_transaction_value'] = round(avg_payment, 2)
        
        print(f"\n✓ Average Transaction Value: ${avg_payment:.2f}")
        
    def calculate_claim_payout_metrics(self):
        """
        KPI: Claim payout metrics
        - Average claim amount
        - Total claims paid
        - Loss ratio (claims paid / premiums collected)
        """
        avg_claim = self.claims['payoutAmount'].mean()
        total_claims_paid = self.claims['payoutAmount'].sum()
        total_premiums = self.policies['premium'].sum()
        
        loss_ratio = (total_claims_paid / total_premiums) * 100 if total_premiums > 0 else 0
        
        self.kpis['average_claim_amount'] = round(avg_claim, 2)
        self.kpis['total_claims_paid'] = round(total_claims_paid, 2)
        self.kpis['loss_ratio_percent'] = round(loss_ratio, 2)
        
        print(f"\n✓ Average Claim Amount: ${avg_claim:.2f}")
        print(f"✓ Total Claims Paid: ${total_claims_paid:.2f}")
        print(f"✓ Loss Ratio: {loss_ratio:.2f}%")
        
    def calculate_time_to_finality(self):
        """
        KPI: Time-to-Finality
        Average time from booking creation to payment confirmation
        Note: Using simulated timestamps; in real system would use blockchain TX times
        """
        # Merge bookings with payments
        merged = self.bookings.merge(
            self.payments, 
            left_on='bookingId', 
            right_on='bookingId',
            how='inner'
        )
        
        # Convert to datetime
        merged['bookingDate'] = pd.to_datetime(merged['bookingDate'])
        merged['paymentDate'] = pd.to_datetime(merged['paymentDate'])
        
        # Calculate time difference in seconds
        merged['time_to_finality_seconds'] = (
            merged['paymentDate'] - merged['bookingDate']
        ).dt.total_seconds()
        
        avg_time = merged['time_to_finality_seconds'].mean()
        median_time = merged['time_to_finality_seconds'].median()
        
        self.kpis['avg_time_to_finality_seconds'] = round(avg_time, 2)
        self.kpis['median_time_to_finality_seconds'] = round(median_time, 2)
        
        print(f"\n✓ Average Time-to-Finality: {avg_time:.2f} seconds ({avg_time/60:.2f} minutes)")
        print(f"✓ Median Time-to-Finality: {median_time:.2f} seconds ({median_time/60:.2f} minutes)")
        
    def calculate_revenue_metrics(self):
        """
        KPI: Revenue metrics
        - Total revenue (payments + premiums)
        - Revenue per user
        """
        total_payment_revenue = self.payments['amount'].sum()
        total_premium_revenue = self.policies['premium'].sum()
        total_revenue = total_payment_revenue + total_premium_revenue
        
        revenue_per_user = total_revenue / len(self.users)
        
        self.kpis['total_revenue'] = round(total_revenue, 2)
        self.kpis['revenue_per_user'] = round(revenue_per_user, 2)
        
        print(f"\n✓ Total Revenue: ${total_revenue:.2f}")
        print(f"✓ Revenue Per User: ${revenue_per_user:.2f}")
        
    def calculate_all_kpis(self):
        """Calculate all business KPIs."""
        self.load_data()
        
        print("\n" + "="*60)
        print("CALCULATING BUSINESS KPIs")
        print("="*60)
        
        self.calculate_conversion_rate()
        self.calculate_policy_purchase_rate()
        self.calculate_claim_rate()
        self.calculate_average_transaction_value()
        self.calculate_claim_payout_metrics()
        self.calculate_time_to_finality()
        self.calculate_revenue_metrics()
        
        # Add metadata
        self.kpis['timestamp'] = datetime.now().isoformat()
        self.kpis['kpi_type'] = 'business'
        
        return self.kpis
    
    def save_kpis(self, output_file='sprint_3/visualizations/business_kpis.json'):
        """Save business KPIs to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.kpis, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ Business KPIs saved to: {output_file}")
        print(f"{'='*60}\n")
        
    def print_summary(self):
        """Print a formatted summary of all KPIs."""
        print("\n" + "="*60)
        print("BUSINESS KPI SUMMARY")
        print("="*60)
        
        print(f"\n📊 CONVERSION & ENGAGEMENT")
        print(f"  • Conversion Rate: {self.kpis.get('conversion_rate_percent', 0):.2f}%")
        print(f"  • Policy Purchase Rate: {self.kpis.get('policy_purchase_rate_percent', 0):.2f}%")
        
        print(f"\n💰 REVENUE METRICS")
        print(f"  • Total Revenue: ${self.kpis.get('total_revenue', 0):,.2f}")
        print(f"  • Revenue Per User: ${self.kpis.get('revenue_per_user', 0):,.2f}")
        print(f"  • Average Transaction Value: ${self.kpis.get('average_transaction_value', 0):,.2f}")
        
        print(f"\n🔔 CLAIMS & RISK")
        print(f"  • Claim Rate: {self.kpis.get('claim_rate_percent', 0):.2f}%")
        print(f"  • Average Claim Amount: ${self.kpis.get('average_claim_amount', 0):,.2f}")
        print(f"  • Loss Ratio: {self.kpis.get('loss_ratio_percent', 0):.2f}%")
        print(f"  • Total Claims Paid: ${self.kpis.get('total_claims_paid', 0):,.2f}")
        
        print(f"\n⚡ PERFORMANCE")
        print(f"  • Avg Time-to-Finality: {self.kpis.get('avg_time_to_finality_seconds', 0):,.2f}s")
        print(f"  • Median Time-to-Finality: {self.kpis.get('median_time_to_finality_seconds', 0):,.2f}s")
        
        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    calculator = BusinessKPICalculator()
    calculator.calculate_all_kpis()
    calculator.print_summary()
    calculator.save_kpis()
