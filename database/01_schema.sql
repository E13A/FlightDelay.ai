-- 01_schema.sql (v4) without description column
DROP MATERIALIZED VIEW IF EXISTS mv_monthly_payouts;
DROP TABLE IF EXISTS InsuranceClaim;
DROP TABLE IF EXISTS Payment;
DROP TABLE IF EXISTS InsurancePolicy;
DROP TABLE IF EXISTS Ticket;
DROP TABLE IF EXISTS Booking;
DROP TABLE IF EXISTS Flight;
DROP TABLE IF EXISTS Developer;
DROP TABLE IF EXISTS "User";
DROP TABLE IF EXISTS StatusLookup;

CREATE TABLE StatusLookup (
    statusId SERIAL PRIMARY KEY,
    statusType VARCHAR(20) NOT NULL,
    code VARCHAR(30) NOT NULL,
    UNIQUE(statusType, code)
);

CREATE TABLE "User"(
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    phone VARCHAR(20)
);

CREATE TABLE Developer(
    developerId SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    role VARCHAR(50),
    responsibility VARCHAR(200)
);

CREATE TABLE Flight(
    flightId SERIAL PRIMARY KEY,
    origin VARCHAR(100) NOT NULL,
    destination VARCHAR(100) NOT NULL,
    departureTime TIMESTAMP NOT NULL,
    arrivalTime TIMESTAMP NOT NULL,
    statusId INT NOT NULL REFERENCES StatusLookup(statusId)
);

CREATE TABLE Booking(
    bookingId SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES "User"(user_id),
    flightId INT NOT NULL REFERENCES Flight(flightId),
    bookingDate TIMESTAMP NOT NULL,
    statusId INT NOT NULL REFERENCES StatusLookup(statusId)
);

CREATE TABLE Ticket(
    ticketId SERIAL PRIMARY KEY,
    bookingId INT NOT NULL REFERENCES Booking(bookingId),
    seatNumber VARCHAR(20) NOT NULL,
    company VARCHAR(100) NOT NULL,
    price NUMERIC(10,2) NOT NULL,
    issueDate TIMESTAMP NOT NULL,
    isPremium BOOLEAN DEFAULT FALSE
);

CREATE TABLE InsurancePolicy(
    policyId SERIAL PRIMARY KEY,
    bookingId INT NOT NULL REFERENCES Booking(bookingId),
    coverageAmount NUMERIC(10,2) NOT NULL,
    premium NUMERIC(10,2) NOT NULL,
    statusId INT NOT NULL REFERENCES StatusLookup(statusId)
);

CREATE TABLE Payment(
    paymentId SERIAL PRIMARY KEY,
    bookingId INT NOT NULL REFERENCES Booking(bookingId),
    amount NUMERIC(10,2) NOT NULL,
    paymentMethod VARCHAR(50) NOT NULL,
    paymentDate TIMESTAMP NOT NULL,
    statusId INT NOT NULL REFERENCES StatusLookup(statusId)
);

CREATE TABLE InsuranceClaim(
    claimId SERIAL PRIMARY KEY,
    policyId INT NOT NULL REFERENCES InsurancePolicy(policyId),
    delayDuration FLOAT NOT NULL,
    claimStatus VARCHAR(50) NOT NULL,
    payoutAmount NUMERIC(10,2) NOT NULL
);
