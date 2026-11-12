Vaani Relay: a rate-limit-aware RAG pipeline
============================================

A Python RAG pipeline over three arXiv papers (1706.03762 v7 "Attention Is All You Need", 2410.00037 v1 "Moshi", 2507.07955 v2 "Dynamic Chunking"). It answers questions using the real Groq free tier (openai/gpt-oss-20b, reasoning_effort="low", include_reasoning=False) and protects a hard token budget while doing it. I also built a calibrated stand-in service for the load tests that the free tier cannot safely take.

This is my submission for the Vaani ML Engineering Intern take-home, "The Relay". Real results and stand-in (simulated) results are kept apart everywhere: separate raw files, a source label in results.csv, and a label on every table row.

I used an AI assistant to draft much of the code and tests. I set the scope and the safety rules, ran everything on my own Mac, read the raw outputs, approved each real run, and decided what to do when something looked wrong. The assistant also wrote a first pass of all 48 grades. A manual review (reviewer not named) later covered 20 of those rows and changed 7. The other 28 rows are still the assistant's first pass: unverified, and not an independent human grading (details below).

Stack
-----

| Part | What I used |
|---|---|
| Language and tests | Python 3, pytest (379 tests, no network and no key needed) |
| Retrieval | BM25 written by hand, word chunks, three settings A, B, C |
| Model | Groq free tier, openai/gpt-oss-20b, reasoning effort low, reasoning not returned |
| Safety | token budget guard with per-stage caps and a global ceiling, atomic JSON ledger |
| Load tests | calibrated stand-in service, six load shapes, four pipeline modes |
| Evidence and output | raw JSONL logs for every request, results.csv (12 columns), tables.md |

How a question flows through it
-------------------------------
PDFs
-> chunking + reference-list filter (app/chunking.py)
-> BM25 retrieval, setting A / B / C (app/retrieval.py)
-> prompt that requires [n] citations (app/answer.py)
-> budget guard reserves worst case (app/budget_guard.py)
-> paced Groq call, max 3 attempts (app/eval_run.py)
-> citation check: were the cited passages really sent? (rag.verify)
-> grades in data/eval/grades.csv: answer grade and citation support, kept separate, with a review column
-> results.csv + Tables 1, 2a, 2b, 2c, 3, 3b, 3c (app/export_csv.py, app/tables.py)

What is in here
---------------

| Path | What |
|---|---|
| app/chunking.py | PDF text to word chunks, reference-list filter |
| app/retrieval.py | hand-written BM25, settings A/B/C, re-rank for C |
| app/answer.py, app/rag.py | prompt, answer, citation check (verify) |
| app/pipeline.py, app/scenarios.py | request pipeline and load shapes (stand-in) |
| app/budget_guard.py, app/eval_budget.py | token budget guard and per-stage caps |
| app/eval_run.py, app/eval_status.py | the controlled real evaluation (Stages A, B, C) |
| app/grading_sheet.py, app/export_csv.py, app/tables.py | grading sheet, results.csv, tables |
| data/eval/questions.csv, grades.csv | my 16 questions and the grades (answer grade, citation support, review status, note) |
| data/eval/archive/ | the original first-pass grades.csv, kept unchanged |
| data/raw/ | raw evidence: every real HTTP attempt, job lines, the ledger, archived originals |
| results.csv, results_detail.csv, tables.md | final outputs |
| docs/ | research, solution, process, results, testing |

Who decided what
----------------

| Decision or step | Who |
|---|---|
| Scope, deadline plan, "no new features, no fake numbers" | me |
| The 16 questions, including the unanswerable Q16 | me |
| Keeping the hard questions (Q04, Q09, Q16) in the set | me |
| No question-specific retrieval tricks | me |
| Token caps 30,000 / 41,000 / 37,000 and ceiling 110,000 | me |
| Revoking and replacing the Groq key, .env in .gitignore | me |
| Approving Stages A, B and C one at a time | me |
| Drafting code, tests and first-pass summaries | assistant, run and read by me |
| First-pass grades for all 48 rows | assistant |
| Manual review of 20 rows (Q01 A, Q10 C, Q11 to Q16 in all settings); it changed 7 rows | reviewer not named; entered into grades.csv with review = manual_review |
| The other 28 rows | assistant's first pass, unverified (review = assistant_first_pass) |

What I found on the real service
--------------------------------

Small probes showed the limits I had to design around: 8,000 tokens per minute (seen in response headers and in three real 429s) and 1,000 requests per day (header only, never exhausted). Eight parallel requests went through with no 429, and a deliberate limit probe got 3 rate-limited out of 8, with Retry-After values of 1, 3 and 5 seconds. Requests per minute and tokens per day were not measured.

The final evaluation was 16 questions in each of three settings, so 48 real requests, run as three stages one after the other. All 48 came back fine, with no retries and no 429. In total 77,806 tokens were used of the 110,000 ceiling.

Table 3: answer grade by retrieval setting (real evaluation, 16 questions each, count and percent of 16)

| Setting | Correct | Partial | Wrong | Refused correctly | Refused wrongly | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| A: 80-word pieces (20 overlap), 8 fetched | 11 (68.8%) | 1 (6.3%) | 1 (6.3%) | 1 (6.3%) | 2 (12.5%) | 1,374 | 780 / 1,346 | 858 / 1,414 |
| B: 200-word pieces (40 overlap), 5 fetched | 12 (75.0%) | 1 (6.3%) | 1 (6.3%) | 1 (6.3%) | 1 (6.3%) | 1,840 | 825 / 1,437 | 953 / 1,606 |
| C: 300-word pieces (60 overlap), 3 fetched, re-rank | 11 (68.8%) | 1 (6.3%) | 2 (12.5%) | 1 (6.3%) | 1 (6.3%) | 1,649 | 733 / 964 | 869 / 1,129 |

Table 3b: citation support, kept separate from the answer grade (count and percent of 16)

| Setting | Manual review / first pass rows | Supported | Partial | Unsupported | Uncited | Refused (no citation) | Correct and supported | Correct, partial citation | Correct, unsupported | Correct, uncited |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 7 / 9 | 12 (75.0%) | 0 (0.0%) | 1 (6.3%) | 0 (0.0%) | 3 (18.8%) | 11 (68.8%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| B | 6 / 10 | 11 (68.8%) | 0 (0.0%) | 2 (12.5%) | 1 (6.3%) | 2 (12.5%) | 10 (62.5%) | 0 (0.0%) | 1 (6.3%) | 1 (6.3%) |
| C | 7 / 9 | 10 (62.5%) | 1 (6.3%) | 3 (18.8%) | 0 (0.0%) | 2 (12.5%) | 9 (56.3%) | 1 (6.3%) | 1 (6.3%) | 0 (0.0%) |

Answer correctness and citation support are two separate judgements. A correct answer with an unsupported, partial or missing citation stays a correct answer and is counted in its own column; it is never turned into a wrong answer. The citation columns above are a mixture of manual-review findings and the assistant's unverified first pass, so they are not a human-verified result. tables.md splits them by review status (Table 3b) and counts the review status of all 48 rows (Table 3c: 20 manual review, 28 assistant first pass).

The settings differ by one question on answer grade (A 11, B 12, C 11 correct of 16), and one question is 6.3 points, so this sample cannot name a winning setting. What did differ was cost and speed: A is cheapest per answer (1,374 tokens against 1,840 for B and 1,649 for C), and C had the tightest first-token p95 (964 ms against 1,346 and 1,437). Among the manually reviewed rows, B has one correct answer without a citation (Q11) and one with an unsupported citation (Q12), C has one unsupported (Q12) and one partial (Q13), and A has none. Q15 is graded partial in all three settings.

Where it failed
---------------

Q02 (why the Transformer divides by the square root of the key dimension) was refused in all three settings, and Q14 in setting A, although the papers answer both. I attribute this to the passage not being retrieved, but I did not check every case. Q12 in setting A cites passages about other methods and is graded wrong; in B and C the Q12 answers are correct, but their cited passages are appendix text that does not state the claim. Q14 in B and C adds claims the cited text does not say. Q09 in C describes the inner monologue wrongly (first-pass grade, not verified). I chose not to add question-specific tricks to force retrieval, because that would improve the numbers without improving the system.

What is still unverified
------------------------

28 of the 48 grades, including all of Q02, are the assistant's first pass and have not been checked by anyone else. grades.csv still has 15 notes starting with CHECK that mark open questions. I did not verify the Q16 passages myself: Q16 is graded correctly refused in all three settings on the strength of the manual review.

What went wrong and how I fixed it
----------------------------------

Four things went wrong in my own measurements. None of the raw evidence was edited: the originals are archived in data/raw/real_archive and the fixed lines say what changed.

| Problem | Cause | Fix |
|---|---|---|
| Reference filter clipped real appendix text | a reference list and an appendix shared one segment | detect where the last entry ends; drop text only if it looks like a bibliography under a References label |
| Stage A showed 4 "late" answers | my own pacing wait (about 56 s) was counted as latency; the service answered in about 1 s | wait recorded separately, Stage A restated, B and C run after the fix |
| Q10 answers marked invalid | the delay pattern [0, 2, 2, 2] was read as a citation of passage 0 | a bracket group that repeats a number is data, not a citation; two rows went from invalid to cited, spend unchanged |
| Misleading diagnostic line | a "bibliography" hit was really an appendix labelled References | I stopped using that line as evidence |

Grading was fixed later too: the first version forced "correct" to mean cited and supported, which could not express a correct answer with a weak citation. The grade vocabulary now has answer grades correct, partial, wrong, refused_correctly and refused_wrongly, and citation support supported, partial, unsupported and uncited, and the two are independent. The original first-pass grades.csv is kept in data/eval/archive/.

About the index: reference-list sections were filtered before chunking, 8,271 words in total (attention 944, moshi 4,033, dynamic_chunking 3,294). Appendix material was not removed and stays indexed. Some appendix passages carry a misleading [References] label, which is how Q12 ended up with appendix text about other methods.

A final check against results.csv also corrected three wrong numbers in earlier drafts: decode time (I had subtracted medians, not per-request values), the stand-in provider 429 count (93, not 96) and the amount of reference text removed.

The stand-in load tests
-----------------------

The stand-in copies the limits I measured and every number from it is simulated. The six load shapes are steady, burst, overload, halved limit, daily limit exhausted and flaky. The four pipeline modes are baseline, a polite baseline that honours Retry-After, controlled without degrading, and controlled.

Percent of requests that succeeded (simulated):

| Scenario | baseline | baseline_polite | controlled_no_degrade | controlled |
|---|---|---|---|---|
| steady | 100 | 100 | 100 | 100 |
| burst | 20 | 35 | 30 | 45 |
| overload | 50 | 52 | 52 | 76 |
| halved_limit | 80 | 86.7 | 80 | 100 |
| daily_exhausted | 53.3 | 53.3 | 53.3 | 53.3 |
| flaky | 100 | 100 | 100 | 100 |

The controlled mode never sends a request that earns a 429, at the price of giving up on some requests that cannot meet their deadline (burst 55%, overload 24%). Refusal origins in the stand-in runs: provider 429 = 93, local deadline shed = 64, local budget gate = 14, late answer = 4. When the daily quota is gone every mode ends at 53.3%, which is right, since no pipeline can beat the quota.

What is not measured
--------------------

Paid-equivalent cost (the repo has no price list), queue length for the real runs (no queue telemetry), requests per minute, tokens per day and the provider's published limits. All of these say "Not measured" in the tables instead of a guess.

Run it (the stand-in and the tests need no key)
-----------------------------------------------

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m pytest -q
python -m app.export_csv && python -m app.tables
```

Real calls are off by default. They run only with --approve-real-calls AND RELAY_APPROVE_REAL=yes AND .env listed in .gitignore. Put your own key in .env as GROQ_API_KEY=... and never commit it.

Headline results (details in docs/results.md)
---------------------------------------------

- 48 real evaluation requests (16 questions x settings A, B, C), all ok, 77,806 tokens of the 110,000 ceiling, no retries, no 429.
- Answer grade: correct A 11, B 12, C 11 of 16, with one partial answer (Q15) in each setting. Citation support is reported separately in Table 3b. 20 of the 48 grades come from a manual review (reviewer not named); the other 28 are unverified first-pass grades. The results are not human-graded.
- Stand-in load tests: the controlled pipeline serves more requests than the polite and plain baselines in burst, overload and halved-limit; every stand-in number is simulated.
