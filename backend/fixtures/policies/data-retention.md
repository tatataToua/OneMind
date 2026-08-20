# Data Retention and Disposal

**Source:** 45 CFR 164.316(b)(2)(i), SOC 2 CC6.5 — internal policy.
**Owner:** Security Office | **Last reviewed:** 2026-06-02

## Retention periods

| Record class | Minimum retention | Maximum retention |
|---|---|---|
| HIPAA policies, procedures, and required documentation | 6 years from creation or last effective date | indefinite |
| Security incident records and investigations | 6 years | indefinite |
| Audit logs of PHI access | 6 years | 7 years |
| Remote patient monitoring device telemetry | 1 year in hot storage | 7 years archived |
| Adjudicated claims records | 7 years | 10 years |
| Model inference request and response logs containing PHI | not retained | 30 days if redacted |

Device telemetry is treated as PHI when it is linked to an identified patient.
The one-year hot-storage window balances clinical utility against exposure;
archived telemetry must be encrypted with a key separate from the hot tier.

## Disposal

Media containing PHI must be sanitised in line with NIST SP 800-88 before
disposal or reuse. Cryptographic erasure is acceptable where the key management
system can attest to key destruction.
