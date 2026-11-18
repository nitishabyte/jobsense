# Relevance labeling: judge and borderline decisions

**Judge:** Claude (LLM). Blind to which search system returned each job and to its score.
**Input:** 166 pooled (query, job) pairs (top-5 from BM25 + top-5 from embeddings, merged and shuffled).
**Result:** 117 relevant, 49 not relevant. Every query has at least 2 relevant jobs in its pool, so under the pre-set rule no query needed replacing.
**Method:** Each job was judged against the query's written `relevant_if` rule. The title was read for every job, and the full description for every borderline case.

## Policies applied consistently
- **Volunteer / intern / contract** postings are judged by the work itself, not by employment type, because no rule excludes them.
- **Maintenance roles that explicitly list HVAC repair among their duties** count as relevant for "fixing heating and air conditioning" (lenient reading of "main duty").
- **Product managers** count only where the rule names a field rather than an engineering role ("cybersecurity / information security role" = yes; "software engineering role" = no).
- **Python listed as one of 2–3 accepted languages** = Python is a main language; **Python listed among 8+ languages** = not a main language.

## Borderline calls

| Query | Job | Label | Reason |
|---|---|---|---|
| python software engineer | Senior Software Engineer (maven), "Python, JavaScript or C++" | 1 | Python is one of three accepted main languages |
| python software engineer | Senior Software Engineer (Elegant Enterprise), lists 9 languages | 0 | Python is not a main language |
| data analyst SQL | Data Science Intern (analysis, SQL) | 1 | Analyst work with SQL; interns not excluded |
| data analyst SQL | Data Analyst, Wells Fargo (Excel, reporting) | 1 | Data analyst using "similar tools" (Excel) |
| data analyst SQL | Systems Analyst (ETL, Informatica, SQL) | 0 | Data-engineering/ETL role, not analysis |
| data analyst SQL | Data Modeler; IT Solutions Analyst; Application Analyst | 0 | Not data/business analyst roles |
| software sales account executive | Account Executive ×2 (building materials) | 0 | Not selling software or technology |
| software sales account executive | AE at NOVUS (requires software-sales experience, product not stated) | 0 | Can't confirm it sells software/tech (strict) |
| software sales account executive | AE at Catalyte (tech talent / software services to enterprises) | 1 | B2B technology services |
| software sales account executive | Territory Sales Rep (selling cybersecurity solutions) | 1 | Technology sold to businesses |
| social media marketing manager | Customer Success Manager at a social media agency (runs client social campaigns) | 1 | Social media campaigns are the core work |
| mechanical engineer | Process Automation Engineer (PLC/controls) | 0 | Controls/automation, not mechanical engineering |
| backend role working with AI | Senior Full-Stack Developer ("primary focus on backend systems") | 1 | Not frontend-only; backend is the primary focus |
| backend role working with AI | Android Developer Intern | 0 | Mobile app, not backend or ML |
| backend role working with AI | Backend Product Manager | 0 | Not an engineering role |
| helping patients recover from injuries | RN Home Health; Home Health LPN; Nurse Case Rep (insurance) | 0 | Not rehabilitation roles |
| keeping a company's books and paying the bills | Finance Administrative Assistant (pays bills, books expenses, reconciles) | 1 | Duties are bookkeeping/AP in substance |
| keeping a company's books and paying the bills | Night Auditor (hotel front desk) | 0 | Hotel front-desk role, not bookkeeping |
| teaching young children | Volunteer music instructor, pre-K to 5th grade | 1 | Teaching young children; volunteers not excluded |
| teaching young children | Kids (Academy) Supervisor at a gym ×2 | 0 | Supervising play and selling programs, not teaching |
| fixing heating and air conditioning | Maintenance Supervisor; Stationary Engineer; Facilities Maintenance (HVAC listed) | 1 | Maintenance policy above |
| fixing heating and air conditioning | HVAC Service Supervisor; HVAC Duct Foreman | 1 | Hands-on HVAC service/installation leadership |
| protecting company computers from hackers | Sr Principal Software Engineer, Palo Alto Networks App Security | 1 | Builds security products |
| protecting company computers from hackers | Cloud Security Product Manager (Global Information Security) | 1 | Security role (PM policy above) |
| protecting company computers from hackers | Armed Security Officer | 0 | Physical security |
| moving boxes in a warehouse | Merchandiser (stocking store shelves) | 0 | Retail store, not a warehouse |
| moving boxes in a warehouse | Warehouse Specialist (forklift license, MHE, inventory) | 1 | Hands-on warehouse work |
| moving boxes in a warehouse | Director Logistics Operations; Customer Success Manager | 0 | Management/office roles |
| cooking food in a restaurant kitchen | Chef de Partie, Whole Foods prepared foods | 1 | Chef role managing food production |
| cooking food in a restaurant kitchen | Seasonal Kitchen Assistant, cooking school (food prep, dishes) | 1 | Kitchen staff doing food prep |
| helping people buy and sell homes | New Home Consultant (Lennar); Community Sales Manager (Taylor Morrison) | 1 | Selling homes directly to buyers (real-estate sales) |
| helping people buy and sell homes | Loan Officer ×2; mortgage admin support | 0 | Excluded by the rule |

Spot check (round 1): 1 answer corrected for a mis-keyed entry before viewing the agreement report. Result: 29/30 agreement, Cohen's κ = 0.93. The one disagreement (n05, job 3902758429) was left with the judge's label.

---

# Round 2 (Step 6: re-ranker)

**Input:** 35 new (query, job) pairs surfaced by `embedding+rerank` that the first two systems had not returned. Same judge, same rules, same policies, still blind to which system found each job.
**Result:** 25 relevant, 10 not relevant. **Total labels after round 2: 201.**

| Query | Job | Label | Reason |
|---|---|---|---|
| tax manager | Resource Manager – Tax (staff scheduling for a tax practice) | 0 | Not a tax accounting role |
| data analyst SQL | Data Analyst (Japanese): data quality analysis with SQL/Excel/Python | 1 | Analyst work with data |
| data analyst SQL | Data Integrity Analyst (ERP data cleansing/governance) | 0 | Data management, consistent with round 1's Systems Analyst / Data Modeler |
| software sales account executive | AE at a B2B marketing/events company; AE selling magazine/media ads | 0 | Not selling software or technology |
| software sales account executive | AE at a Microsoft Solutions Partner | 1 | Selling technology services |
| backend role working with AI | "AI/Generative AI Positions" (openings incl. AI/ML engineers) | 1 | ML/AI engineering roles |
| talking to customers on the phone all day | Receptionist, H&R Block ×2 | 0 | Consistent with round 1 |
| talking to customers on the phone all day | Technical Sales (4–5 hours of call-center-style cold calling per day) | 1 | Phone conversations are the core of the job |
| fixing heating and air conditioning | HVAC retail sales manager; HVAC water-treatment chemical testing | 0 | Not HVAC repair/installation |
| protecting company computers from hackers | Warehouse Coordinator at a cybersecurity company | 0 | Warehouse role |
| cooking food in a restaurant kitchen | Kitchen Assistant, Sur La Table | 1 | Consistent with round 1 |
| helping people buy and sell homes | Community Manager, affordable housing (property management) | 0 | Property management excluded |
