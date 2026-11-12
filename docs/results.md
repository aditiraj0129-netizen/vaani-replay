Results
=======

Real evaluation, 16 questions in each setting (all 48 requests succeeded, 77,806 tokens used of the 110,000 ceiling). Counts are out of 16, with the percent in brackets.

Setting A, 80-word pieces, 8 fetched: 11 correct (68.8%), 1 partial (6.3%), 1 wrong (6.3%), 1 correctly refused (6.3%), 2 refused wrongly (12.5%), 1,374 tokens per answer, first token p50 780 ms and p95 1,346 ms, total time p50 858 ms and p95 1,414 ms.

Setting B, 200-word pieces, 5 fetched: 12 correct (75.0%), 1 partial, 1 wrong, 1 correctly refused, 1 refused wrongly (6.3% each), 1,840 tokens per answer, first token 825 and 1,437 ms, total time 953 and 1,606 ms.

Setting C, 300-word pieces, 3 fetched with re-ranking: 11 correct (68.8%), 1 partial (6.3%), 2 wrong (12.5%), 1 correctly refused and 1 refused wrongly (6.3% each), 1,649 tokens per answer, first token 733 and 964 ms, total time 869 and 1,129 ms.

Citation support is a separate judgement from the answer grade. Of the correct answers: A has 11 with a supported citation; B has 10 supported, 1 with an unsupported citation (Q12) and 1 with no citation (Q11); C has 9 supported, 1 with an unsupported citation (Q12) and 1 with a partial citation (Q13). These stay correct answers and are not counted as wrong. Q15 is a partial answer in all three settings.

How the grades were made: all 48 started as the assistant's first pass. A manual review (reviewer not named) covered 20 rows and changed 7. The other 28 rows, including all of Q02, are unverified first-pass grades. The results are not human-graded, and the citation numbers above mix both kinds of rows. tables.md has the split (Table 3b and Table 3c). "Wrong" and "refused wrongly" are separate columns; an answer refused when the paper did hold the answer is "refused wrongly".

The settings differ by one question on answer grade (A 11, B 12, C 11), and one question is 6.3 points, so this sample cannot name a winning setting. They differ on cost and speed. With 16 questions the sample is small.

Simulated load tests, percent of requests that succeeded.
steady: 100 in every mode.
burst: baseline 20, polite 35, controlled without degrade 30, controlled 45.
overload: 50, 52, 52, 76.
halved limit: 80, 86.7, 80, 100.
daily limit exhausted: 53.3 in every mode, which is the quota ceiling.
flaky: 100 in every mode.

Controlled mode never sends a request that gets a 429, but it gives up on some that miss their deadline (burst 55%, overload 24%). Reasons for refusals in the stand-in runs: provider 429 93 (stand-in; the 3 real 429s in the limit probe are counted separately), local deadline shed 64, local budget gate 14, late answer 4.

Not measured: paid-equivalent cost (no prices file) and queue length for real runs.
