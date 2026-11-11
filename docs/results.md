Results
=======

Real evaluation, 16 questions in each setting (all 48 requests succeeded, 77,806 tokens used of the 110,000 ceiling).

Setting A, 80-word pieces, 8 fetched: 75.0% correct and cited, 18.8% wrong, 6.2% correctly refused, 1,374 tokens per answer, first token p50 780 ms and p95 1,346 ms, total time p50 858 ms and p95 1,414 ms.

Setting B, 200-word pieces, 5 fetched: 75.0%, 18.8%, 6.2%, 1,840 tokens per answer, first token 825 and 1,437 ms, total time 953 and 1,606 ms.

Setting C, 300-word pieces, 3 fetched with re-ranking: 75.0%, 18.8%, 6.2%, 1,649 tokens per answer, first token 733 and 964 ms, total time 869 and 1,129 ms.

"Wrong" includes answers refused when the paper did hold the answer. The settings tie on quality here and differ on cost and speed. With 16 questions the sample is small.

Simulated load tests, percent of requests that succeeded.
steady: 100 in every mode.
burst: baseline 20, polite 35, controlled without degrade 30, controlled 45.
overload: 50, 52, 52, 76.
halved limit: 80, 86.7, 80, 100.
daily limit exhausted: 53.3 in every mode, which is the quota ceiling.
flaky: 100 in every mode.

Controlled mode never sends a request that gets a 429, but it gives up on some that miss their deadline (burst 55%, overload 24%). Reasons for refusals in the stand-in runs: provider 429 93 (stand-in; the 3 real 429s in the limit probe are counted separately), local deadline shed 64, local budget gate 14, late answer 4.

Not measured: paid-equivalent cost (no prices file) and queue length for real runs.
