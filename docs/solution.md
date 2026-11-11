How the solution works
======================

The papers are turned into word chunks, filtered for reference lists, and indexed with a hand-written BM25. For each question the top chunks go into a prompt that requires citations like [1]. A checker confirms every cited number is a passage that was really sent, and a human review in grades.csv judges whether the passage supports the answer. An answer counts as correct only if both agree.

The budget guard sits in front of every real call. It reserves the worst case, refuses before a stage cap or the global ceiling is passed, treats a 429 as fully spent, and keeps an atomic ledger so a crash cannot hide spent tokens. The real runner sends one request at a time, paced to 85 percent of the per-minute limit, with at most three attempts per question, and it resumes without repeating finished jobs.

For load tests that the free tier cannot safely take, a stand-in service copies the measured limits and runs six load shapes through four pipeline modes. All stand-in numbers are labelled simulated.

I kept the design small on purpose. When the deadline got close I decided against new features and against redesigning anything.
