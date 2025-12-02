-- 03_indexes.sql (v4)
CREATE INDEX IF NOT EXISTS idx_booking_user ON Booking(user_id);
CREATE INDEX IF NOT EXISTS idx_booking_flight ON Booking(flightId);
CREATE INDEX IF NOT EXISTS idx_ticket_booking ON Ticket(bookingId);
CREATE INDEX IF NOT EXISTS idx_policy_booking ON InsurancePolicy(bookingId);
CREATE INDEX IF NOT EXISTS idx_payment_booking ON Payment(bookingId);
CREATE INDEX IF NOT EXISTS idx_claim_policy ON InsuranceClaim(policyId);
CREATE INDEX IF NOT EXISTS idx_payment_date ON Payment(paymentDate);
CREATE INDEX IF NOT EXISTS idx_flight_destination ON Flight(destination);
