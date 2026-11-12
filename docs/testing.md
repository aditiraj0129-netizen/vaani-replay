Testing
=======

python -m pytest -q runs 379 tests with no network and no key. They cover chunking and the reference filter, retrieval, the citation checker, the budget guard including crash recovery and the global ceiling, the real runner's safety switches, the stand-in service, the exports and tables, and the two measurement fixes. export_csv also validates results.csv (12 columns, no guessed statuses) every time it runs, and checks that the answer grades and citation-support values in grades.csv use the allowed vocabulary and do not contradict the pipeline's own citation check. Tables 3, 3b and 3c are checked for counts, percentages (rounded half up) and the review-status split.
