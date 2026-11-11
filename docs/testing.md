Testing
=======

python -m pytest -q runs 375 tests with no network and no key. They cover chunking and the reference filter, retrieval, the citation checker, the budget guard including crash recovery and the global ceiling, the real runner's safety switches, the stand-in service, the exports and tables, and the two measurement fixes. export_csv also validates results.csv (12 columns, no guessed statuses) every time it runs.
