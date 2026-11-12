Process, failures and decisions
===============================

I worked with an AI assistant on this. It drafted code and tests and I ran them, read the results and decided what happened next. I did not let it change raw evidence. When something was wrong, the original file was archived and a restated line was written next to it.

Decisions I made. I kept all 16 questions even though some (Q04, Q09, Q16) were awkward, since dropping hard ones would flatter the results. I refused question-specific retrieval fixes. I set the stage caps (30,000, 41,000, 37,000) and the 110,000 ceiling and told the assistant to stop the moment the guard tripped, an unexpected error appeared or the ledger looked wrong. Close to the deadline I froze new features. Before the real run I revoked the old API key and put a new one in .env, with .env in .gitignore, and only then approved the stages one at a time.

What went wrong and how it was fixed.
1. The first reference filter clipped real appendix words. I changed it to find where the last entry ends and to drop text only when it really looks like a bibliography.
2. Stage A showed four "late" answers. The cause was my own pacing wait being counted as latency, about 56 seconds, against a real service latency near one second. The runner now records the wait separately, Stage A was restated, and B and C ran after the fix.
3. The citation checker treated the delay pattern [0, 2, 2, 2] in Q10 as a citation. A citation never repeats a number, so such groups are now ignored. Two rows moved from invalid to cited.
4. A diagnostic line about bibliography pieces misled me, because the hit I blamed was an appendix mislabelled as references. I stopped using it.
5. The first grading format could not express a correct answer with a weak citation, so it forced such answers to be graded wrong. The vocabulary now keeps answer grade and citation support independent, and grades.csv has a review column.

What I could not fix. Q02 was refused in all three settings and Q14 in setting A although the papers answer them. I left it and wrote it down.

Grading. The assistant drafted a first set of grades for all 48 real requests, and that file is archived unchanged in data/eval/archive/. A manual review (reviewer not named) then covered 20 rows (Q01 A, Q10 C, and Q11 to Q16 in all three settings) and changed 7 of them: Q11 B (correct, no citation), Q12 B and C (correct, unsupported citation), Q13 C (correct, partial citation) and Q15 A, B and C (partial answers). The other 28 rows, including all of Q02, are still the assistant's first pass: unverified, and not an independent human grading. 15 notes starting with CHECK mark open questions on first-pass rows. I did not verify the Q16 passages myself. Equal-looking scores across the three settings are real on this sample, but one question moves a result by 6.3 points, so I name no winner.
