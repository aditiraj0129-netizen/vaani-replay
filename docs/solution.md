How the solution works
======================

The papers are turned into word chunks, filtered for reference lists, and indexed with a hand-written BM25. For each question the top chunks go into a prompt that requires citations like [1]. A checker confirms every cited number is a passage that was really sent. That is a mechanical check ("provenance"), and it does not say whether the passage backs the answer. Two judgements are then kept separate in grades.csv: the answer grade (correct, partial, wrong, refused correctly, refused wrongly) and the citation support (supported, partial, unsupported, uncited). A correct answer can carry a weak or missing citation, and it is reported that way instead of being turned into a wrong answer. A review column says whether a row came from the manual review (20 rows, reviewer not named) or is the assistant's unverified first pass (28 rows).

The budget guard sits in front of every real call. It reserves the worst case, refuses before a stage cap or the global ceiling is passed, treats a 429 as fully spent, and keeps an atomic ledger so a crash cannot hide spent tokens. The real runner sends one request at a time, paced to 85 percent of the per-minute limit, with at most three attempts per question, and it resumes without repeating finished jobs.

For load tests that the free tier cannot safely take, a stand-in service copies the measured limits and runs six load shapes through four pipeline modes. All stand-in numbers are labelled simulated.

I kept the design small on purpose. When the deadline got close I decided against new features and against redesigning anything.
