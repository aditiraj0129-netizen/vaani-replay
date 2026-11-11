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
