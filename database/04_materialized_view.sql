-- 04_materialized_view.sql (v4)
DROP MATERIALIZED VIEW IF EXISTS mv_monthly_payouts;

CREATE MATERIALIZED VIEW mv_monthly_payouts AS
SELECT
    f.destination,
    date_trunc('month', p.paymentDate) AS month,
    SUM(ic.payoutAmount) AS total_payout
FROM InsuranceClaim ic
JOIN InsurancePolicy ip ON ip.policyId = ic.policyId
JOIN Booking b ON b.bookingId = ip.bookingId
JOIN Flight f ON f.flightId = b.flightId
JOIN Payment p ON p.bookingId = b.bookingId
GROUP BY f.destination, date_trunc('month', p.paymentDate);

CREATE INDEX IF NOT EXISTS idx_mv_month_dest ON mv_monthly_payouts(destination);
