Evaluated on 20 queries (10 title-style, 10 natural-language), labels: 166 judged (query, job) pairs.

| System | P@5 (all) | P@5 title-style | P@5 natural-language | MRR@5 | Median latency |
|---|---|---|---|---|---|
| bm25 | 0.680 | 0.860 | 0.500 | 0.814 | 8.1 ms |
| embedding | 0.800 | 0.920 | 0.680 | 0.896 | 14.7 ms |

Per-query precision@5:

| Query | Type | bm25 | embedding |
|---|---|---|---|
| registered nurse | title | 1.0 | 1.0 |
| python software engineer | title | 0.8 | 0.8 |
| tax manager | title | 1.0 | 1.0 |
| CDL truck driver | title | 1.0 | 1.0 |
| data analyst SQL | title | 0.2 | 1.0 |
| software sales account executive | title | 0.8 | 0.4 |
| physical therapist | title | 1.0 | 1.0 |
| construction project manager | title | 1.0 | 1.0 |
| social media marketing manager | title | 1.0 | 1.0 |
| mechanical engineer | title | 0.8 | 1.0 |
| backend role working with AI | natural | 0.6 | 0.4 |
| helping patients recover from injuries | natural | 0.2 | 0.4 |
| talking to customers on the phone all day | natural | 0.2 | 0.6 |
| keeping a company's books and paying the bills | natural | 0.4 | 0.8 |
| teaching young children | natural | 0.8 | 0.6 |
| fixing heating and air conditioning | natural | 1.0 | 1.0 |
| protecting company computers from hackers | natural | 0.0 | 1.0 |
| moving boxes in a warehouse | natural | 0.4 | 0.8 |
| cooking food in a restaurant kitchen | natural | 1.0 | 1.0 |
| helping people buy and sell homes | natural | 0.4 | 0.2 |

embedding vs bm25: better on 7 queries, worse on 4, tied on 9.
