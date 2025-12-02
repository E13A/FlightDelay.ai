import unittest
import pandas as pd
import sys
import os

# Add parent directory to path to import ds_pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ds_pipeline import FeatureEngineer

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.engineer = FeatureEngineer()
        
        # Mock Data
        self.users = pd.DataFrame({
            'user_id': [1, 2],
            'name': ['Alice', 'Bob']
        })
        
        self.bookings = pd.DataFrame({
            'bookingId': [101, 102],
            'user_id': [1, 2],
            'flightId': [1001, 1002],
            'bookingDate': ['2023-01-01', '2023-01-02'],
            'statusId': [1, 1]
        })
        
        self.payments = pd.DataFrame({
            'paymentId': [201, 202],
            'bookingId': [101, 102],
            'amount': [100.0, 200.0],
            'paymentDate': ['2023-01-01 10:00:00', '2023-01-02 10:00:00']
        })
        
        # Mocking Claims linked via Policy (assuming logic)
        # We need policies to link claims to bookings if that's the schema
        self.policies = pd.DataFrame({
            'policyId': [301, 302],
            'bookingId': [101, 102]
        })
        
        self.claims = pd.DataFrame({
            'claimId': [401],
            'policyId': [301],
            'payoutAmount': [50.0]
        })
        
        self.flights = pd.DataFrame({
            'flightId': [1001, 1002],
            'departureTime': ['2023-01-10', '2023-01-12']
        })

    def test_process_basic(self):
        # Process the data
        df = self.engineer.process(self.users, self.bookings, self.payments, self.claims, self.flights, self.policies)
        
        # Assertions
        self.assertEqual(len(df), 2) # Should have 2 bookings
        
        # Check Booking 101 (User 1)
        # Payment: 100
        # Claim: 50 (via Policy 301)
        # Ratio: 0.5
        row1 = df[df['bookingId'] == 101].iloc[0]
        self.assertEqual(row1['total_payment'], 100.0)
        self.assertEqual(row1['total_claim_amount'], 50.0)
        self.assertEqual(row1['claim_ratio'], 0.5)
        
        # Check Booking 102 (User 2)
        # Payment: 200
        # Claim: 0 (No claim)
        # Ratio: 0.0
        row2 = df[df['bookingId'] == 102].iloc[0]
        self.assertEqual(row2['total_payment'], 200.0)
        self.assertEqual(row2['total_claim_amount'], 0.0)
        self.assertEqual(row2['claim_ratio'], 0.0)
        
        # Check Days to Departure
        # Booking 101: 2023-01-01 -> 2023-01-10 = 9 days
        self.assertEqual(row1['days_to_departure'], 9)

if __name__ == '__main__':
    unittest.main()
