DPDP Act Q&A: RAG with Section-Level Citations

A retrieval-augmented question-answering system over India's Digital Personal Data Protection Act, 2023. Ask a question in plain language and get an answer that cites the sections it came from. It says so when the Act does not cover the question.

How it works
Ingest: extract text from the Act's PDF with pypdf.
Section-aware chunking: split at section boundaries (sections accepted in order 1 to 44, which avoids false splits from stray numbered lines).
Sub-chunking: sections longer than 1,500 characters are cut into overlapping pieces, because the embedding model reads only the start of long inputs.
Embed and index: BAAI/bge-small-en-v1.5 embeddings, normalized, stored in a FAISS inner-product index.
Small-to-big retrieval: search over sub-chunks for precision, then pass the full parent sections of the top hits to the LLM so answers are not truncated.
Generate: Gemini answers only from the supplied sections, cites section numbers, and refuses when the context lacks the answer. Includes retry and model fallback for API overload (503/429).
Demo: Gradio interface showing the answer and the sections used.
Evaluation

31 hand-written questions, each labelled with the section that answers it. Metric: hit@3, whether the correct section appears among the top 3 retrieved results.

Chunking strategy	hit@3
Fixed-size (800 chars, 100 overlap)	58.1%
Section-aware	83.9%
Section-aware + sub-chunks	90.3%

With sub-chunks, the correct section is ranked first for 87.1% of questions (k=1). Increasing k from 3 to 5 added nothing, so k=3 is used.

An extra experiment, adding a section-context header to every later sub-chunk, scored 92.3% (29 of 31), one question better than 90.3%. With only 31 questions that difference is within noise, so I kept the simpler sub-chunking in the demo.

Out-of-scope questions (tax evasion, a cricket result) were correctly refused, 2 of 2 tested.

What I learned
My first section splitter was wrong. It only detected sections that began with a capital letter, so it found 14 of 44 sections and gave a misleading 19% hit@3. Checking the chunk count and section list caught it. Fixing the regex and enforcing in-order section numbers gave 44 clean chunks.
Stale indexes give silently wrong results. After changing the chunks, the old index was still in memory and retrieval returned unrelated sections. Rebuilding all indexes after any chunking change fixed it.
Retrieving only the matched sub-chunk truncated answers. A question about processing without consent returned a list cut off after two of nine clauses. Small-to-big retrieval (search small, generate from the full section) produced the complete list.
Audit the eval labels. After a miss, I checked the PDF and found one of my own labels was wrong, so I corrected it. A miss can mean the label is wrong, not only that retrieval failed.
Failure analysis (remaining misses)
What must a fiduciary tell a person when asking for consent? (expected section 5, notice) retrieved section 6 (consent). The two sections overlap heavily in wording.
Is consent needed during a medical emergency? (expected section 7) retrieved sections 6 and 9. The emergency clause sits deep inside a long section and its sub-chunk reads like generic consent text.
Can a person with a physical disability be part of the Board? (expected section 21) retrieved sections 14, 9 and 2. Section 14 (nominating someone on incapacity) uses similar words.

All three are cases of wording overlap between related sections.

Limitations
Small eval set: one question is about 3 percentage points.
Labels are section-level and written by me, and five questions are concentrated on the Board (sections 18 to 22).
The fixed-size baseline labels each window by the section where it starts, so it may slightly understate that baseline.
Answer quality was spot-checked, not scored systematically.
Not legal advice.
Run it
bash
pip install -r requirements.txt

Open dpdp_rag.ipynb (built in Google Colab), upload the Act's PDF, and add a Gemini API key as a secret named GEMINI_API_KEY. Run the cells in order. If you change the chunking, rerun everything from the indexing cell onward.

Stack

Python, pypdf, sentence-transformers, FAISS, Google Gemini API, Gradio
