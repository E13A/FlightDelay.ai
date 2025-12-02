-- 02_seed.sql (v4)
INSERT INTO StatusLookup(statusId,statusType,code) VALUES
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
(22,'payment','Disputed');

INSERT INTO "User"(name,email,phone) VALUES
('Alice','alice@example.com','111-111'),
('Bob','bob@example.com','222-222');

INSERT INTO Developer(name,role,responsibility) VALUES
('Sara Dev','Backend','Flight & claim verification'),
('John Fin','Finance','Payment validation & exchange rates');

INSERT INTO Flight(origin,destination,departureTime,arrivalTime,statusId) VALUES
('Baku','Istanbul','2025-01-10 10:00','2025-01-10 12:30',1),
('Istanbul','London','2025-01-11 14:00','2025-01-11 17:30',1);

INSERT INTO Booking(user_id,flightId,bookingDate,statusId) VALUES
(1,1,NOW(),8),
(2,2,NOW(),8);

INSERT INTO Ticket(bookingId,seatNumber,company,price,issueDate,isPremium) VALUES
(1,'12A','Caspian Air',250,NOW(),TRUE),
(2,'16C','EuroSky Airlines',430,NOW(),FALSE);

INSERT INTO InsurancePolicy(bookingId,coverageAmount,premium,statusId) VALUES
(1,1000,25,13),
(2,1500,35,13);

INSERT INTO Payment(bookingId,amount,paymentMethod,paymentDate,statusId) VALUES
(1,275,'Credit Card',NOW(),18),
(2,465,'PayPal',NOW(),18);

INSERT INTO InsuranceClaim(policyId,delayDuration,claimStatus,payoutAmount) VALUES
(1,3.5,'Approved',200),
(2,5,'Pending',250);
