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

How I worked with the assistant
-------------------------------

The assistant drafted code, tests and first-pass summaries. I set the limits (token caps, the ceiling, no question-specific retrieval hacks, no invented measurements, no new architecture near the deadline), asked for the fixes above when the numbers looked wrong, revoked and replaced my API key before the real run, and approved each real stage myself. The assistant's draft grades were a starting point for my own review, not the final word. docs/process.md has the longer story.

Where things are
----------------

app holds the code, tests holds the tests, data/eval has the questions, grading sheet and grades, and data/raw has every raw request and the budget ledger. results.csv, results_detail.csv and tables.md are the outputs. docs has the longer notes.
