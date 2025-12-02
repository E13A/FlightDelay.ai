# Data Retention Policy

## Overview
This document defines the data retention policy for the Flight Insurance DApp, covering on-chain and off-chain data storage, retention periods, and archival procedures.

## Data Classification

### On-Chain Data
- **Smart Contract State**: Permanent (immutable on blockchain)
- **Transaction Records**: Permanent (blockchain history)
- **Policy NFT Metadata**: Permanent (IPFS/blockchain storage)

### Off-Chain Data
- **User Information**: Retained for 7 years (regulatory compliance)
- **Flight Data**: Retained for 2 years
- **Booking Records**: Retained for 3 years
- **Payment Records**: Retained for 7 years (financial compliance)
- **Insurance Claims**: Retained for 10 years (legal requirements)
- **Event Logs**: Retained for 1 year
- **Analytics/Aggregated Data**: Retained for 5 years

## Archival Strategy

### Hot Storage (Active)
- **Duration**: Last 6 months
- **Location**: Primary PostgreSQL database
- **Access**: Real-time queries, full indexing
- **Use Cases**: Dashboard, model inputs, live analytics

### Warm Storage (Recent Archive)
- **Duration**: 6 months - 3 years
- **Location**: Secondary database or partitioned tables
- **Access**: Reduced indexing, slower queries acceptable
- **Use Cases**: Historical reporting, compliance queries

### Cold Storage (Long-term Archive)
- **Duration**: 3+ years
- **Location**: Compressed files in object storage (S3/IPFS)
- **Format**: Parquet or CSV snapshots
- **Access**: Manual retrieval, batch processing only
- **Use Cases**: Regulatory audits, legal discovery

## Archival Procedures

### Monthly Archive Process
1. **Identify Records**: Select records older than 6 months from hot storage
2. **Validation**: Verify data integrity before archival
3. **Export**: Generate compressed snapshots (Parquet format)
4. **Storage**: Upload to cold storage with metadata tags
5. **Cleanup**: Remove archived records from hot storage
6. **Verification**: Confirm successful archival and accessibility

### Materialized View Refresh
- **Frequency**: Daily at 2 AM UTC
- **Command**: `REFRESH MATERIALIZED VIEW mv_monthly_payouts;`
- **Retention**: Aggregated data retained for 5 years
- **Archival**: Monthly snapshots exported to cold storage

## Data Deletion Policy

### Personal Data (GDPR Compliance)
- **Right to Erasure**: Users can request data deletion after contract obligations expire
- **Grace Period**: 30 days after policy expiration
- **Exceptions**: Records required for legal/regulatory compliance are pseudonymized instead

### System Logs
- **Application Logs**: Delete after 90 days
- **Error Logs**: Delete after 1 year
- **Audit Logs**: Retain for 7 years

## Implementation

### Automated Scripts
- `sprint_3/archival_script.py`: Monthly archival automation
- Cron job: `0 2 1 * * /path/to/archival_script.py`

### Monitoring
- Alert if archival fails
- Dashboard showing hot/warm/cold storage utilization
- Compliance report generation quarterly
