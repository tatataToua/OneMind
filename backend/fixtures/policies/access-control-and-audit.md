# Access Control, Audit Logging, and Minimum Necessary

**Source:** 45 CFR 164.308(a)(4), 164.312(a)-(b), 164.502(b), ISO 27001 A.9.
**Owner:** Security Office | **Last reviewed:** 2026-06-02

## Minimum necessary

Requests for PHI must be limited to the minimum necessary to accomplish the
intended purpose. Access is granted by role, and roles are reviewed quarterly.
The minimum necessary standard does not apply to disclosures to the individual
who is the subject of the information, nor to treatment-related disclosures
between providers.

## Automated systems acting on behalf of a user

Where an automated system retrieves PHI on behalf of a user, the system inherits
that user's authorisation and must not broaden it. A component able to reach data
the requesting user could not reach directly violates minimum necessary,
regardless of whether that data ultimately reaches the user.

## Audit logging

Log all access to PHI with actor, timestamp, resource, and the purpose asserted.
Audit logs themselves must not contain PHI values — reference records by
identifier, never by content. Logs must be tamper-evident and reviewed on a
defined cadence.
