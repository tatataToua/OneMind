# De-identification of Protected Health Information

**Source:** 45 CFR 164.514(a)-(b) — paraphrased for internal guidance.
**Owner:** Privacy Office | **Last reviewed:** 2026-04-11

Health information is de-identified when it does not identify an individual and
there is no reasonable basis to believe it can be used to identify one. Two
methods are permitted.

## Safe Harbor method

Remove all eighteen identifiers of the individual and of relatives, employers,
and household members, including: names; geographic subdivisions smaller than a
state; all date elements more precise than year, including birth date, admission
date, discharge date, and date of death; telephone and fax numbers; email
addresses; Social Security numbers; medical record numbers; health plan
beneficiary numbers; account numbers; certificate and licence numbers; vehicle
and device identifiers and serial numbers; URLs and IP addresses; biometric
identifiers; full-face photographs; and any other unique identifying number,
characteristic, or code.

Ages over 89 must be aggregated into a single category of 90 or older.

## Expert Determination method

A person with appropriate statistical and scientific knowledge documents that
the risk of re-identification is very small, alone or in combination with other
reasonably available information.

## Internal standard

Systems handling PHI must redact identifiers before transmitting content to any
external inference endpoint. Where a reversible token map is maintained in order
to restore identifiers in output returned to an authorised user, that map must
never leave the trust boundary and must be discarded at end of session.
