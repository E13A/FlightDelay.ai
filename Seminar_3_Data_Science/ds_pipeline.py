import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_csv(self, filename):
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{filename} not found in {self.data_dir}")
        return pd.read_csv(path)

    def load_all(self):
        users = self.load_csv('User.csv')
        bookings = self.load_csv('Booking.csv')
        payments = self.load_csv('Payment.csv')
        claims = self.load_csv('InsuranceClaim.csv')
        flights = self.load_csv('Flight.csv')
        policies = self.load_csv('InsurancePolicy.csv')
        return users, bookings, payments, claims, flights, policies

class FeatureEngineer:
    def __init__(self):
        pass

    def process(self, users, bookings, payments, claims, flights, policies):
        # Merge Data
        # bookings -> users
        df = bookings.merge(users, on='user_id', how='left')
        
        # payments -> bookings
        # Payment.csv: paymentId,bookingId,amount,paymentMethod,paymentDate,statusId
        # Note: 'amount' might be string with currency symbol, need to clean if necessary. 
        # Assuming numeric for now based on typical CSVs, but good to be safe.
        payments_agg = payments.groupby('bookingId').agg({
            'amount': 'sum',
            'paymentDate': 'max'
        }).reset_index().rename(columns={'amount': 'total_payment', 'paymentDate': 'last_payment_date'})
        
        df = df.merge(payments_agg, left_on='bookingId', right_on='bookingId', how='left')
        
        # claims -> policies -> bookings
        # InsuranceClaim.csv: claimId,policyId,delayDuration,claimStatus,payout
        # InsurancePolicy.csv: policyId,bookingId,...
        
        claims_with_policy = claims.merge(policies[['policyId', 'bookingId']], on='policyId', how='left')
        
        claims_agg = claims_with_policy.groupby('bookingId').agg({
            'payoutAmount': 'sum',
            'claimId': 'count'
        }).reset_index().rename(columns={'payoutAmount': 'total_claim_amount', 'claimId': 'claim_count'})
        
        df = df.merge(claims_agg, left_on='bookingId', right_on='bookingId', how='left')
        
        # flights -> bookings
        # Flight.csv: flightId,flightNumber,origin,destination,departureTime,arrivalTime,price
        df = df.merge(flights, left_on='flightId', right_on='flightId', how='left')
        
        # Feature Engineering
        df['total_payment'] = df['total_payment'].fillna(0)
        df['total_claim_amount'] = df['total_claim_amount'].fillna(0)
        df['claim_count'] = df['claim_count'].fillna(0)
        
        # Derived Features
        # 1. Claim Ratio (Claim Amount / Payment Amount)
        df['claim_ratio'] = df.apply(lambda x: x['total_claim_amount'] / x['total_payment'] if x['total_payment'] > 0 else 0, axis=1)
        
        # 2. Booking Lag (Time between booking and departure)
        if 'bookingDate' in df.columns and 'departureTime' in df.columns:
            df['bookingDate'] = pd.to_datetime(df['bookingDate'], errors='coerce')
            df['departureTime'] = pd.to_datetime(df['departureTime'], errors='coerce')
            df['days_to_departure'] = (df['departureTime'] - df['bookingDate']).dt.days
        
        return df

if __name__ == "__main__":
    # Example usage
    loader = DataLoader('.')
    users, bookings, payments, claims, flights, policies = loader.load_all()
    engineer = FeatureEngineer()
    df_features = engineer.process(users, bookings, payments, claims, flights, policies)
    print(df_features.head())
    df_features.to_csv('sprint_3/features.csv', index=False)
