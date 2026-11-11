
Vaani Relay
===========

This is my submission for the Vaani ML Engineering Intern take-home, "The Relay". It is a small RAG pipeline that answers questions about three arXiv papers (Attention Is All You Need, Moshi, and the Dynamic Chunking paper), calls the real Groq free tier, and tries hard not to waste or overspend tokens while doing it. I also built a stand-in service that behaves like Groq's free tier, so I could test burst and overload traffic without burning my real quota.

I used an AI assistant to draft a lot of the code. I decided the scope, wrote the evaluation questions, ran everything on my own Mac, read the outputs, approved every real run, and made the calls when something looked wrong. The honest account of that is at the bottom and in docs/process.md.

What it does
------------

The three PDFs are cut into word chunks and searched with a BM25 retriever I kept hand-written, so I can explain every line of it. The top chunks go into a prompt for openai/gpt-oss-20b (reasoning_effort low, reasoning not returned). The model must cite passages like [1], and a small checker then confirms that every cited number really was a passage we sent. Whether the cited passage actually supports the answer is a separate judgement, and that one is mine, written into data/eval/grades.csv.

Every real request first reserves its worst-case tokens in a budget guard. If the answer would push a stage over its cap, or the whole run over the ceiling, the guard refuses before the call happens. A 429 or an unknown result is counted as fully spent. The real runner will not even start unless I pass a flag, set an environment variable, and .env is listed in .gitignore.

Three retrieval settings were compared: A (80-word pieces, 8 fetched), B (200-word pieces, 5 fetched) and C (300-word pieces, 3 fetched after re-ranking 12 candidates). I wrote 16 questions, including one that the papers cannot answer (Q16) to see whether the system refuses properly.

What I found on the real service
--------------------------------

Small probes showed the limits I had to design around: about 8,000 tokens per minute and 1,000 requests per day. Eight parallel requests went through with no 429, while a deliberate limit probe got 3 rate-limited out of 8. Retry-After values were 1, 3 and 5 seconds.

The final evaluation was 16 questions in each of the three settings, so 48 real requests, run as three stages one after the other with caps of 30,000, 41,000 and 37,000 tokens and an overall ceiling of 110,000. All 48 requests came back fine, with no retries and no 429. In total 77,806 tokens were used, which left 32,194 unspent.

On quality, all three settings landed in the same place: 12 of 16 correct and properly cited, 3 wrong or wrongly refused, and 1 correct refusal. I did not expect a perfect tie and I do not want to oversell it. With only 16 questions, one question moves a setting by 6.2 points, so I would not claim one setting is better from this. What did differ was cost and speed: A used about 1,374 tokens per answer, C about 1,649, and B about 1,840. C had the tightest first-token and total-time spread.

Where it failed
---------------

The failures were mostly retrieval, not the model. Q02 (why the Transformer divides by the square root of the key dimension) was refused in all three settings because the right passage was never retrieved. Q06, Q07, Q11, Q12 and Q13 also had retrieval misses in every setting. I tried removing reference lists from the index, which was the obvious suspect, and it ran correctly but did not change which questions fail. I decided not to add question-specific tricks to force retrieval. That would have made the numbers look better without making the system better.

A few answers were wrong even though the right topic was retrieved. For example, Q09 in C described the inner monologue wrongly, and Q14 in B and C added claims that the cited text does not say. Those are marked in the grades file with a short note.

What I fixed along the way
--------------------------

Four things went wrong in my own measurements, and I fixed each without touching the raw evidence. The originals are kept in data/raw/real_archive and the fixed lines say what changed.

First, the reference-list filter. My first version clipped real appendix text where a reference list and an appendix shared a segment. I changed it to detect where the last entry ends, and it now only drops text that looks like a bibliography under a References label. About 5,000 words of reference text per paper set were removed. A small leak of around 60 words at the start of two lists remains, and I left it rather than risk deleting real content.

Second, pacing. Stage A showed four answers marked "late". That was my own pacing sleep (about 56 seconds, used to stay under the token-per-minute limit) being counted as latency. The service itself had answered in about one second. I fixed the runner so the wait is recorded separately, restated those 16 lines of Stage A, and ran B and C with the fix in place.

Third, a citation checker bug. A Q10 answer contained a delay pattern, [0, 2, 2, 2], and the checker read it as a citation of passage 0 and marked the answer invalid. A real citation never repeats a number, so a bracket group that does is now treated as data. Two rows changed from invalid to cited. Tokens and spend did not change.

Fourth, one diagnostic line I had built ("bibliography pieces among sent") turned out to be misleading, because a hit I thought was a bibliography was actually an appendix mislabelled as References. I stopped relying on that line.

The stand-in load tests
-----------------------

The stand-in is calibrated to the limits I measured, and every number from it is simulated. I kept six load shapes (steady, burst, overload, halved limit, daily limit exhausted, flaky) and four pipeline modes (baseline, a polite baseline that honours Retry-After, controlled without degrading, and controlled). In burst, overload and halved-limit traffic the controlled mode served the most requests (for example 76% against 50% in overload) and never sent a request that earned a 429, at the price of giving up on some requests that could not meet their deadline. When the daily limit is gone, every mode ends at the same 53.3%, which is correct, because no pipeline can do better than the quota. Real and simulated rows are labelled separately in results.csv and in every table.

What is not measured
--------------------

I did not measure paid-equivalent cost because the repository has no price list, and I did not record queue length for the real runs because there is no queue telemetry there. Both say "Not measured" in the tables instead of a guess.

Running it
----------

Create a virtual environment, install requirements.txt, and run python -m pytest -q (375 tests, no network and no key needed). python -m app.export_csv and python -m app.tables rebuild results.csv and the tables from the raw data. Real calls need your own key in .env as GROQ_API_KEY and the three safety switches described above. Never commit the key.


Where things are
----------------

app holds the code, tests holds the tests, data/eval has the questions, grading sheet and grades, and data/raw has every raw request and the budget ledger. results.csv, results_detail.csv and tables.md are the outputs. docs has the longer notes.
EOF_README
cat > docs/research.md <<'EOF_R'
Research notes
==============

Before writing the pipeline I wanted three answers: what the Groq free tier really allows for this model, how to keep a hard token budget while still retrying, and which chunk size works for papers as different as a Transformer paper and a speech-model paper.

For the first, I ran small, cheap probes and read the response headers. They showed roughly 8,000 tokens per minute and 1,000 requests per day. Eight requests at once caused no 429, and a deliberate limit probe produced 3 rate-limited out of 8. Retry-After came back as 1, 3 or 5 seconds.

For the second, I decided every request must reserve its worst case before it is sent, so retries cannot sneak past a cap. That became the budget guard.

For the third, I compared three settings (80, 200 and 300 word pieces) on 16 questions I wrote myself. The papers also contain long reference lists and appendices, which can pollute retrieval, so I wrote a filter for them and measured what it removed: 944 words from the Attention paper, 4,033 from Moshi and 3,294 from the Dynamic Chunking paper. It did not change which questions fail, which told me the misses come from wording and not from bibliography noise.
EOF_R
cat > docs/solution.md <<'EOF_S'
How the solution works
======================

The papers are turned into word chunks, filtered for reference lists, and indexed with a hand-written BM25. For each question the top chunks go into a prompt that requires citations like [1]. A checker confirms every cited number is a passage that was really sent, and a human review in grades.csv judges whether the passage supports the answer. An answer counts as correct only if both agree.

The budget guard sits in front of every real call. It reserves the worst case, refuses before a stage cap or the global ceiling is passed, treats a 429 as fully spent, and keeps an atomic ledger so a crash cannot hide spent tokens. The real runner sends one request at a time, paced to 85 percent of the per-minute limit, with at most three attempts per question, and it resumes without repeating finished jobs.

For load tests that the free tier cannot safely take, a stand-in service copies the measured limits and runs six load shapes through four pipeline modes. All stand-in numbers are labelled simulated.


Process, failures and decisions
===============================

I worked with an AI assistant on this. It drafted code and tests and I ran them, read the results and decided what happened next. I did not let it change raw evidence. When something was wrong, the original file was archived and a restated line was written next to it.

Decisions I made. I kept all 16 questions even though some (Q04, Q09, Q16) were awkward, since dropping hard ones would flatter the results. I refused question-specific retrieval fixes. I set the stage caps (30,000, 41,000, 37,000) and the 110,000 ceiling and told the assistant to stop the moment the guard tripped, an unexpected error appeared or the ledger looked wrong. Close to the deadline I froze new features. Before the real run I revoked the old API key and put a new one in .env, with .env in .gitignore, and only then approved the stages one at a time.

What went wrong and how it was fixed.
1. The first reference filter clipped real appendix words. I changed it to find where the last entry ends and to drop text only when it really looks like a bibliography.
2. Stage A showed four "late" answers. The cause was my own pacing wait being counted as latency, about 56 seconds, against a real service latency near one second. The runner now records the wait separately, Stage A was restated, and B and C ran after the fix.
3. The citation checker treated the delay pattern [0, 2, 2, 2] in Q10 as a citation. A citation never repeats a number, so such groups are now ignored. Two rows moved from invalid to cited.
4. A diagnostic line about bibliography pieces misled me, because the hit I blamed was an appendix mislabelled as references. I stopped using it.

What I could not fix. Retrieval still misses Q02, Q06, Q07, Q11, Q12 and Q13 in every setting. I left it and wrote it down.

Grading. The assistant drafted a first set of grades. Those were only a draft. I went through them against the cited passages and the grades in data/eval/grades.csv are the ones I stand behind. Equal scores across the three settings are real on this sample, and I say so in the write-up instead of hiding it.
EOF_P
cat > docs/results.md <<'EOF_RES'
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

Controlled mode never sends a request that gets a 429, but it gives up on some that miss their deadline (burst 55%, overload 24%). Reasons for refusals in the stand-in runs: provider 429 96, local deadline shed 64, local budget gate 14, late answer 4.


Testing
=======

python -m pytest -q runs 375 tests with no network and no key. They cover chunking and the reference filter, retrieval, the citation checker, the budget guard including crash recovery and the global ceiling, the real runner's safety switches, the stand-in service, the exports and tables, and the two measurement fixes. export_csv also validates results.csv (12 columns, no guessed statuses) every time it runs.
EOF_T
grep -c '^#' README.md docs/*.md; ls docs
A Python RAG pipeline over three arXiv papers (1706.03762 v7 "Attention Is All You Need", 2410.00037 v1 "Moshi", 2507.07955 v2 "Dynamic Chunking"). It answers questions using the real Groq free tier (openai/gpt-oss-20b, reasoning_effort="low", include_reasoning=False) and protects a hard token budget while doing it. I also built a calibrated stand-in service for the load tests that the free tier cannot safely take.

This is my submission for the Vaani ML Engineering Intern take-home, "The Relay". Real results and stand-in (simulated) results are kept apart everywhere: separate raw files, a source label in results.csv, and a label on every table row.

 I set the scope, wrote the 16 evaluation questions, ran everything on my own Mac, read every output, approved each real run, and decided what to do when something looked wrong. The honest account is near the end and in docs/process.md.

Stack
-----

| Part | What I used |
|---|---|
| Language and tests | Python 3, pytest (375 tests, no network and no key needed) |
| Retrieval | BM25 written by hand, word chunks, three settings A, B, C |
| Model | Groq free tier, openai/gpt-oss-20b, reasoning effort low, reasoning not returned |
| Safety | token budget guard with per-stage caps and a global ceiling, atomic JSON ledger |
| Load tests | calibrated stand-in service, six load shapes, four pipeline modes |
| Evidence and output | raw JSONL logs for every request, results.csv (12 columns), tables.md |

How a question flows through it
-------------------------------

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
| data/eval/questions.csv, grades.csv | my 16 questions and the grades |
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
| Drafting code, tests and first-pass summaries | assistant, reviewed and run by me |
| First draft of the grades | assistant, then checked by me against the cited passages |
| Final grades in grades.csv | me |

What I found on the real service
--------------------------------

Small probes showed the limits I had to design around: about 8,000 tokens per minute and 1,000 requests per day. Eight parallel requests went through with no 429, and a deliberate limit probe got 3 rate-limited out of 8. Retry-After values were 1, 3 and 5 seconds.

The final evaluation was 16 questions in each of three settings, so 48 real requests, run as three stages one after the other. All 48 came back fine, with no retries and no 429. In total 77,806 tokens were used of the 110,000 ceiling.

Table 3: retrieval settings (real evaluation, 16 questions each)

| Setting | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| A: 80-word pieces (20 overlap), 8 fetched | 75.0 | 18.8 | 6.2 | 1,374 | 780 / 1,346 | 858 / 1,414 |
| B: 200-word pieces (40 overlap), 5 fetched | 75.0 | 18.8 | 6.2 | 1,840 | 825 / 1,437 | 953 / 1,606 |
| C: 300-word pieces (60 overlap), 3 fetched, re-rank | 75.0 | 18.8 | 6.2 | 1,649 | 733 / 964 | 869 / 1,129 |

All three settings landed in the same place on quality. I did not expect a perfect tie and I will not oversell it. With 16 questions, one question moves a setting by 6.2 points, so this sample cannot say one setting is better. What did differ was cost and speed: A is cheapest per answer, B the most expensive, and C had the tightest timing spread.

Where it failed
---------------

The failures were mostly retrieval, not the model. Q02 (why the Transformer divides by the square root of the key dimension) was refused in all three settings because the right passage was never retrieved. Q06, Q07, Q11, Q12 and Q13 also had retrieval misses everywhere. I removed the reference lists from the index, which was the obvious suspect, and it worked as intended but did not change which questions fail. I chose not to add question-specific tricks to force retrieval, because that would improve the numbers without improving the system.

A few answers were wrong even with the right topic retrieved. Q09 in C described the inner monologue wrongly, and Q14 in B and C added claims the cited text does not say. These are marked with short notes in grades.csv.

What went wrong and how I fixed it
----------------------------------

Four things went wrong in my own measurements. None of the raw evidence was edited: the originals are archived in data/raw/real_archive and the fixed lines say what changed.

| Problem | Cause | Fix |
|---|---|---|
| Reference filter clipped real appendix text | a reference list and an appendix shared one segment | detect where the last entry ends; drop text only if it looks like a bibliography under a References label |
| Stage A showed 4 "late" answers | my own pacing wait (about 56 s) was counted as latency; the service answered in about 1 s | wait recorded separately, Stage A restated, B and C run after the fix |
| Q10 answers marked invalid | the delay pattern [0, 2, 2, 2] was read as a citation of passage 0 | a bracket group that repeats a number is data, not a citation; two rows went from invalid to cited, spend unchanged |
| Misleading diagnostic line | a "bibliography" hit was really an appendix labelled References | I stopped using that line as evidence |

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

The controlled mode never sends a request that earns a 429, at the price of giving up on some requests that cannot meet their deadline (burst 55%, overload 24%). When the daily quota is gone every mode ends at 53.3%, which is right, since no pipeline can beat the quota.

What is not measured
--------------------

Paid-equivalent cost (the repo has no price list) and queue length for the real runs (no queue telemetry). Both say "Not measured" in the tables instead of a guess.

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
- Human-graded quality: 12 of 16 correct and cited in each setting; retrieval wording, not chunk size, caused most misses.
- Stand-in load tests: the controlled pipeline serves more requests than the polite and plain baselines in burst, overload and halved-limit; every stand-in number is simulated.
EOF_README
cat > docs/results.md <<'EOF_RES'
Results
=======

Real evaluation, 16 questions in each setting. All 48 requests succeeded, 77,806 tokens used of the 110,000 ceiling (A 21,978, B 29,448, C 26,380).

| Setting | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| A: 80-word pieces, 8 fetched | 75.0 | 18.8 | 6.2 | 1,374 | 780 / 1,346 | 858 / 1,414 |
| B: 200-word pieces, 5 fetched | 75.0 | 18.8 | 6.2 | 1,840 | 825 / 1,437 | 953 / 1,606 |
| C: 300-word pieces, 3 fetched, re-rank | 75.0 | 18.8 | 6.2 | 1,649 | 733 / 964 | 869 / 1,129 |

"Wrong" counts wrong answers and answers refused when the paper did hold the answer. The settings tie on quality and differ on cost and speed. Sixteen questions is a small sample.

Real service probes (not simulated)

| Probe | Result |
|---|---|
| concurrency | 15 of 15 ok, up to 8 at once, no 429 |
| latency | 12 of 12 ok |
| limit | 5 ok, 3 rate-limited (37.5%) |
| reservation | 4 of 4 ok |
| smoke | 1 of 1 ok |

Stand-in load tests (SIMULATED), percent of requests that succeeded

| Scenario | baseline | baseline_polite | controlled_no_degrade | controlled |
|---|---|---|---|---|
| steady | 100 | 100 | 100 | 100 |
| burst | 20 | 35 | 30 | 45 |
| overload | 50 | 52 | 52 | 76 |
| halved_limit | 80 | 86.7 | 80 | 100 |
| daily_exhausted | 53.3 | 53.3 | 53.3 | 53.3 |
| flaky | 100 | 100 | 100 | 100 |

Reasons for refusals in the stand-in runs: provider 429 = 96, local deadline shed = 64, local budget gate = 14, late answer = 4.

Not measured: paid-equivalent cost (no prices in data/eval/pricing.json) and queue length for the real runs.
EOF_RES
grep -c '^#' README.md docs/*.md; wc -w README.md
