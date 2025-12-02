# EXPLAIN Plans and Query Optimisation (v4)

This version uses a unified StatusLookup table without a description column.
The logical relational algebra of the main queries is unchanged compared to
previous versions; only the way status values are stored has been simplified.

## 1. Get all bookings for a given user

**Query:**

```sql
EXPLAIN ANALYZE
SELECT *
FROM Booking
WHERE user_id = 1;
```

**What to look for:**

- After running `03_indexes.sql`, PostgreSQL should use:
  - `Index Scan` on `Booking` using `idx_booking_user`
- Without the index, it would use a `Seq Scan` (full table scan).

## 2. Get all claims for a given booking

**Query:**

```sql
EXPLAIN ANALYZE
SELECT ic.*
FROM InsuranceClaim ic
JOIN InsurancePolicy ip ON ip.policyId = ic.policyId
JOIN Booking b          ON b.bookingId = ip.bookingId
WHERE b.bookingId = 1;
```

**Plan characteristics:**

- Index Scan on `InsuranceClaim(policyId)` using `idx_claim_policy`.
- Index Scan or Nested Loop join with `InsurancePolicy(bookingId)` using `idx_policy_booking`.
- Primary key lookup on `Booking(bookingId)`.

The planner may choose Nested Loop or Hash Join depending on data size, but
the important point is that join columns are indexed.

## 3. Dashboard query behind the materialized view

**Query:**

```sql
EXPLAIN ANALYZE
SELECT
    f.destination,
    date_trunc('month', p.paymentDate) AS month,
    SUM(ic.payoutAmount) AS total_payout
FROM InsuranceClaim ic
JOIN InsurancePolicy ip ON ip.policyId = ic.policyId
JOIN Booking b          ON b.bookingId = ip.bookingId
JOIN Flight  f          ON f.flightId = b.flightId
JOIN Payment p          ON p.bookingId = b.bookingId
GROUP BY f.destination, date_trunc('month', p.paymentDate);
```

This is the same query used in `04_materialized_view.sql` for
`mv_monthly_payouts`. The view precomputes this aggregation so dashboards
can just do:

```sql
SELECT *
FROM mv_monthly_payouts
ORDER BY month, destination;
```

An index on `(destination)` inside the materialized view further speeds up
filtered dashboard queries.

## 4. Relational Algebra and Normalisation

- The unified StatusLookup table:

  ```text
  StatusLookup(statusId, statusType, code)
  ```

  holds all status codes for flights, bookings, policies, and payments.
- Main tables store only integer foreign keys (`statusId`), which removes
  repeated text and enforces consistent status usage.
- Relational algebra expressions for the core business queries are the same
  as in earlier documentation (selections σ, projections π, joins ⋈, and
  grouping γ). The only difference is which table we join to when we want
  the symbolic status code:

  ```sql
  SELECT b.bookingId, s.code AS booking_status
  FROM Booking b
  JOIN StatusLookup s ON s.statusId = b.statusId
  WHERE s.statusType = 'booking';
  ```

  This corresponds to a join between Booking and StatusLookup plus a
  selection on `statusType = 'booking'`.

