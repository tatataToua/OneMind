# OneData Software Solutions — research notes

Background gathered before design, to ground the build in their actual business
rather than a generic agent demo. Sources are listed at the end.

## Company

| | |
|---|---|
| Name | OneData Software Solutions |
| Tagline | "Tech Made Easy for Your Everyday Business" |
| Founded | 2015 |
| Scale | ~150 technical experts, 500+ clients |
| US offices | Fort Mill SC, Atlanta GA |
| Other offices | Coimbatore (India), Toronto, Querétaro, Bogotá, Ratmalana |
| Certifications | ISO 27001, SOC 2 Type II, HIPAA, GDPR-ready |

## Why this matters to the build

**They are an AWS Advanced Tier Services Partner, and their GenAI practice runs
on Bedrock with Anthropic Claude.** Their OneCare case study — a GenAI SDLC
toolkit for healthcare application development — used Claude 3.5 Sonnet on
Bedrock with S3, Secrets Manager, and CloudWatch. Their wider AI stack spans
SageMaker, Lex, Polly, Textract, Comprehend, Kendra, and Fraud Detector.

*Build consequence:* `llm/bedrock.py` exists so "how does this go to
production?" is a one-environment-variable answer rather than a hand-wave. The
provider protocol was designed around that swap, not retrofitted to it.

**They already sell AI agents.** Their AWS AI Agent offering describes a
three-stage loop in their own words:

> goal determination → information acquisition → task execution, tracking
> progress and adapting based on feedback

*Build consequence:* the pipeline deliberately echoes that shape — route
(determine the goal and who owns it), dispatch (acquire information from the
owning data plane), synthesise (execute and report). Using a prospective
employer's own vocabulary back at them is cheap and lands.

**Healthcare is their lead vertical.** Their healthcare line covers EHR,
Patient Relationship Management, healthcare analytics, population health,
Revenue Cycle Management, healthcare IoT, and telehealth / remote patient
monitoring.

*Build consequence:* the four specialists map onto that line directly —

| Specialist | OneData service line |
|---|---|
| Clinical | EHR / care coordination |
| Revenue Cycle | Revenue Cycle Management |
| Compliance | HIPAA / ISO 27001 / SOC 2 posture |
| Remote Monitoring | Healthcare IoT + telehealth RPM |

Population Health and Patient Relationship Management were considered and cut —
see [decisions.md](decisions.md#1-four-specialists-split-by-data-plane-rather-than-by-topic)
for why mapping every service line would have made routing worse, not better.

## Service lines (full)

Data analytics and BI · cloud (AWS consulting, migration, security, managed
services) · IoT development · custom software · web development · ERP and CRM ·
UI/UX · digital marketing.

## Industries

Healthcare, manufacturing, retail, logistics, energy, agriculture, education,
fintech.

## Named products and case studies

OneOps, ERPONE, VesselLog, OneCare, ChatOne. Public metrics cited on their site
include a 99.98% uptime SLA and 98.7% model accuracy.

## Partnerships

AWS Advanced Tier Services Partner. ThingsBoard Silver Partner (IoT).

---

### Sources

- <https://www.onedatasoftware.com/>
- <https://offerings.onedatasoftware.com/aws-ai-agent/>
- <https://offerings.onedatasoftware.com/aws-generative-ai-services/>
- <https://www.onedatasoftware.com/case-study/onecare-revolutionizing-healthcare-application-development-with-genai-sdlc-toolkit>
- <https://www.onedatasoftware.com/industries/healthcare>
- <https://clutch.co/profile/onedata-software-solutions>
- <https://www.linkedin.com/company/onedata-software-solutions>
