"""Generate synthetic data for Sprint 3.

Produces CSV files compatible with the schema in SmartContracts_ETL_DB/database.
"""
import argparse
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def make_statuslookup():
    rows = [
        (1,'flight','Scheduled'),
        (2,'flight','On-Time'),
        (3,'flight','Delayed'),
        (4,'flight','Cancelled'),
        (5,'flight','Departed'),
        (6,'flight','Arrived'),
        (7,'booking','Pending'),
        (8,'booking','Confirmed'),
        (9,'booking','Cancelled'),
        (10,'booking','Completed'),
        (11,'booking','Failed'),
        (12,'policy','Pending'),
        (13,'policy','Active'),
        (14,'policy','Expired'),
        (15,'policy','Claimed'),
        (16,'policy','Closed'),
        (17,'payment','Pending'),
        (18,'payment','Completed'),
        (19,'payment','Failed'),
        (20,'payment','Refunded'),
        (21,'payment','Processing'),
        (22,'payment','Disputed')
    ]
    return pd.DataFrame(rows, columns=['statusId','statusType','code'])


def generate(n_bookings=5000, out_dir='output', seed=42):
    seed_everything(seed)
    fake = Faker()
    os.makedirs(out_dir, exist_ok=True)

    # Users
    n_users = max(1000, n_bookings // 4)
    users = []
    for i in range(1, n_users+1):
        users.append((i, fake.name(), f'user{i}@example.com', fake.phone_number()))
    df_users = pd.DataFrame(users, columns=['user_id','name','email','phone'])

    # Developers (small static set)
    developers = [(1,'Sara Dev','Backend','Flight & claim verification'),
                  (2,'John Fin','Finance','Payment validation & exchange rates')]
    df_devs = pd.DataFrame(developers, columns=['developerId','name','role','responsibility'])

    # Flights
    n_flights = max(300, n_bookings // 10)
    cities = ['Baku','Istanbul','London','Paris','Berlin','Rome','Madrid','Dubai','New York','Toronto']
    flights = []
    base_date = datetime.now() - timedelta(days=365)
    for fid in range(1, n_flights+1):
        origin = random.choice(cities)
        dest = random.choice([c for c in cities if c != origin])
        dep = base_date + timedelta(days=random.randint(0, 365), hours=random.randint(0,23), minutes=random.randint(0,59))
        arr = dep + timedelta(hours=random.randint(1,8), minutes=random.randint(0,59))
        statusId = random.choices([1,2,3,5,6], weights=[0.1,0.6,0.15,0.1,0.05])[0]
        flights.append((fid, origin, dest, dep.isoformat(' '), arr.isoformat(' '), statusId))
    df_flights = pd.DataFrame(flights, columns=['flightId','origin','destination','departureTime','arrivalTime','statusId'])

    # Bookings
    bookings = []
    for bid in range(1, n_bookings+1):
        user_id = random.randint(1, n_users)
        flightId = random.randint(1, n_flights)
        bookingDate = base_date + timedelta(days=random.randint(0, 365), hours=random.randint(0,23))
        statusId = random.choices([7,8,9,10,11], weights=[0.02,0.8,0.03,0.12,0.03])[0]
        bookings.append((bid, user_id, flightId, bookingDate.isoformat(' '), statusId))
    df_bookings = pd.DataFrame(bookings, columns=['bookingId','user_id','flightId','bookingDate','statusId'])

    # Tickets
    companies = ['Caspian Air','EuroSky Airlines','OpenWings','TransGlobal']
    tickets = []
    for bid in range(1, n_bookings+1):
        seat = f"{random.randint(1,40)}{random.choice(['A','B','C','D','E','F'])}"
        company = random.choice(companies)
        price = round(random.uniform(50, 800), 2)
        issueDate = (datetime.fromisoformat(df_bookings.loc[bid-1,'bookingDate']) - timedelta(days=random.randint(0,10))).isoformat(' ')
        isPremium = random.random() < 0.12
        tickets.append((bid, bid, seat, company, price, issueDate, isPremium))
    df_tickets = pd.DataFrame(tickets, columns=['ticketId','bookingId','seatNumber','company','price','issueDate','isPremium'])

    # Insurance policies (e.g., ~60% of bookings buy insurance)
    policies = []
    policy_id = 1
    for booking_row in df_bookings.itertuples():
        if random.random() < 0.6:
            coverage = round(random.uniform(500, 3000),2)
            premium = round(coverage * random.uniform(0.02, 0.08),2)
            statusId = 13  # Active
            policies.append((policy_id, booking_row.bookingId, coverage, premium, statusId))
            policy_id += 1
    df_policies = pd.DataFrame(policies, columns=['policyId','bookingId','coverageAmount','premium','statusId'])

    # Payments
    payments = []
    pay_id = 1
    for booking_row in df_bookings.itertuples():
        amount = round(df_tickets.loc[booking_row.bookingId-1,'price'] * (1 + (0.0)),2)
        method = random.choice(['Credit Card','PayPal','Crypto','Bank Transfer'])
        paymentDate = (datetime.fromisoformat(booking_row.bookingDate) + timedelta(hours=random.randint(0,72))).isoformat(' ')
        statusId = random.choices([17,18,19,20,21,22], weights=[0.01,0.9,0.01,0.03,0.03,0.02])[0]
        payments.append((pay_id, booking_row.bookingId, amount, method, paymentDate, statusId))
        pay_id += 1
    df_payments = pd.DataFrame(payments, columns=['paymentId','bookingId','amount','paymentMethod','paymentDate','statusId'])

    # Insurance claims: subset of policies -> claim
    claims = []
    claim_id = 1
    for p in df_policies.itertuples():
        if random.random() < 0.12:  # ~12% of policies lead to claims
            delay = round(np.random.exponential(scale=2.5) + random.choice([0,1,2]),2)
            claimStatus = random.choices(['Approved','Pending','Rejected'], weights=[0.6,0.3,0.1])[0]
            payout = round(min(p.coverageAmount, delay * random.uniform(30,120)),2)
            claims.append((claim_id, p.policyId, delay, claimStatus, payout))
            claim_id += 1
    df_claims = pd.DataFrame(claims, columns=['claimId','policyId','delayDuration','claimStatus','payoutAmount'])

    # Logging/model trace and MAB assignment
    # Simple heuristic: risk_score = function of price, isPremium, recent_claims_by_user
    # For reproducibility, compute number of prior claims per user from generated claims
    # Map policy -> user via booking
    policy_user = df_policies.merge(df_bookings[['bookingId','user_id']], on='bookingId')
    policy_user = policy_user[['policyId','user_id']]
    claim_policy = df_claims[['policyId']].copy()
    claim_user = claim_policy.merge(policy_user, on='policyId', how='left') if not claim_policy.empty else pd.DataFrame(columns=['policyId','user_id'])

    prior_claims = claim_user['user_id'].value_counts().to_dict()

    mab_rows = []
    trace_rows = []
    treatments = ['baseline','variant_a','variant_b']
    for b in df_bookings.itertuples():
        user_prior = prior_claims.get(b.user_id, 0)
        ticket = df_tickets.loc[b.bookingId-1]
        price = ticket['price']
        isPremium = bool(ticket['isPremium'])
        # simple risk score
        risk_score = min(1.0, (price / 800) * 0.6 + (0.2 if isPremium else 0.0) + min(0.4, user_prior * 0.15))

        # epsilon-greedy assignment emulation
        eps = 0.1
        if random.random() < eps:
            assigned = random.choice(treatments)
        else:
            assigned = 'baseline' if risk_score < 0.5 else random.choice(['variant_a','variant_b'])

        mab_rows.append((b.bookingId, b.user_id, assigned))

        trace_rows.append((b.bookingId, b.user_id, price, isPremium, user_prior, risk_score, datetime.utcnow().isoformat(' ')))

    df_mab = pd.DataFrame(mab_rows, columns=['bookingId','user_id','treatment'])
    df_trace = pd.DataFrame(trace_rows, columns=['bookingId','user_id','price','isPremium','prior_claims','risk_score','timestamp'])

    # Materialized view CSV (monthly payouts by destination)
    if not df_claims.empty:
        mv = df_claims.merge(df_policies[['policyId','bookingId']], on='policyId')
        mv = mv.merge(df_bookings[['bookingId','flightId']], on='bookingId')
        mv = mv.merge(df_flights[['flightId','destination']], on='flightId')
        mv = mv.merge(df_payments[['bookingId','paymentDate']], on='bookingId')
        mv['month'] = pd.to_datetime(mv['paymentDate']).dt.to_period('M').dt.to_timestamp()
        mv_monthly = mv.groupby(['destination','month']).agg(total_payout=('payoutAmount','sum')).reset_index()
    else:
        mv_monthly = pd.DataFrame(columns=['destination','month','total_payout'])

    # Write CSVs
    df_status = make_statuslookup()
    df_status.to_csv(os.path.join(out_dir,'StatusLookup.csv'), index=False)
    df_users.to_csv(os.path.join(out_dir,'User.csv'), index=False)
    df_devs.to_csv(os.path.join(out_dir,'Developer.csv'), index=False)
    df_flights.to_csv(os.path.join(out_dir,'Flight.csv'), index=False)
    df_bookings.to_csv(os.path.join(out_dir,'Booking.csv'), index=False)
    df_tickets.to_csv(os.path.join(out_dir,'Ticket.csv'), index=False)
    df_policies.to_csv(os.path.join(out_dir,'InsurancePolicy.csv'), index=False)
    df_payments.to_csv(os.path.join(out_dir,'Payment.csv'), index=False)
    df_claims.to_csv(os.path.join(out_dir,'InsuranceClaim.csv'), index=False)
    df_mab.to_csv(os.path.join(out_dir,'mab_assignments.csv'), index=False)
    df_trace.to_csv(os.path.join(out_dir,'model_trace.csv'), index=False)
    mv_monthly.to_csv(os.path.join(out_dir,'mv_monthly_payouts.csv'), index=False)

    # Create a simple SQL loader file for Postgres using COPY (user should adjust paths)
    sql_lines = []
    csvs = ['StatusLookup','User','Developer','Flight','Booking','Ticket','InsurancePolicy','Payment','InsuranceClaim']
    pwd = os.getcwd()
    for name in csvs:
        path = os.path.join(pwd, out_dir, f"{name}.csv").replace('\\','\\\\')
        sql_lines.append(f"\copy {name} FROM '{path}' WITH CSV HEADER;")

    with open(os.path.join(out_dir,'populate_data.sql'),'w', encoding='utf-8') as fh:
        fh.write('\n'.join(sql_lines))

    print(f"Wrote CSVs to {out_dir}; total bookings={len(df_bookings)}; policies={len(df_policies)}; claims={len(df_claims)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=5000, help='Number of bookings to generate')
    p.add_argument('--out-dir', default=os.path.join(os.path.dirname(__file__), 'output'))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    generate(n_bookings=args.n, out_dir=args.out_dir, seed=args.seed)


if __name__ == '__main__':
    main()
