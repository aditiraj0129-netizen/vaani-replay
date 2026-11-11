Research notes
==============

Before writing the pipeline I wanted three answers: what the Groq free tier really allows for this model, how to keep a hard token budget while still retrying, and which chunk size works for papers as different as a Transformer paper and a speech-model paper.

For the first, I ran small, cheap probes and read the response headers. They showed roughly 8,000 tokens per minute and 1,000 requests per day. Eight requests at once caused no 429, and a deliberate limit probe produced 3 rate-limited out of 8. Retry-After came back as 1, 3 or 5 seconds.

For the second, I decided every request must reserve its worst case before it is sent, so retries cannot sneak past a cap. That became the budget guard.

For the third, I compared three settings (80, 200 and 300 word pieces) on 16 questions I wrote myself. The papers also contain long reference lists and appendices, which can pollute retrieval, so I wrote a filter for them and measured what it removed: 944 words from the Attention paper, 4,033 from Moshi and 3,294 from the Dynamic Chunking paper. It did not change which questions fail, which told me the misses come from wording and not from bibliography noise.
