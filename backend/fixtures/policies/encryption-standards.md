# Encryption Standards

**Source:** 45 CFR 164.312(a)(2)(iv), 164.312(e), SOC 2 CC6.1, ISO 27001 A.10.
**Owner:** Security Office | **Last reviewed:** 2026-06-02

## In transit

All PHI in transit must use TLS 1.2 or higher with forward secrecy. TLS 1.0 and
1.1 are prohibited. Internal service-to-service traffic crossing a host boundary
is in scope, including traffic to inference endpoints.

## At rest

PHI at rest must be encrypted with AES-256 or equivalent. Keys are managed in a
dedicated key management service with rotation at most every 365 days.
Application code must never hold long-lived key material.

## Local and on-premises inference

Where model inference runs on hardware under our physical control and no PHI
crosses a network boundary, the in-transit requirement is satisfied by the
absence of transmission. This does not relieve the at-rest requirement for any
cached prompts, embeddings, or logs, which remain in scope and must not persist
PHI in plaintext.
