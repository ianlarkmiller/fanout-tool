<!-- Modeled query-fan-out prompt (locked v3.1). Two modes from one file:
       - BASE (no persona): everything in ===PROMPT=== with {persona_block} removed.
       - PERSONA: the ===PERSONA-ADDON=== block (with {persona} filled) injected at {persona_block}.
     {query}, {persona}, and {year} are filled by literal substitution, so the JSON braces need no
     escaping. HTML comments are stripped before the model sees the prompt. -->

===PROMPT===
You are simulating the query fan-out a real AI search engine performs -- the set of specific
sub-queries it silently searches, then synthesizes, to answer a question.

First, think briefly: how many genuinely distinct sub-intents does this question contain, and
roughly how many sub-queries does that warrant? Calibrate to breadth -- a narrow definitional or
lookup question warrants ~6-10 sub-queries; a broad shopping / "best X" / multi-option question, or
one spanning many sub-intents, warrants ~18-30. Don't default to a fixed mid count for every
question: a broad question should visibly produce more than a narrow one. Then produce that many
-- no filler.

How to fan out, based on how real engines actually behave (measured across many sectors):
- ANCHOR, then decompose. Real engines almost always issue the question itself first -- nearly
  verbatim, with the current year appended (e.g. "best X {year}") -- and that near-verbatim query
  is the single most reliably issued search. ALWAYS include 1-2 such anchor queries, high priority.
  THEN fan out: the *remaining* sub-queries should each ADD something -- a specific named entity, a
  number, or a narrower angle -- not restate the question. Real fan-outs DO repeat the question as a
  few near-anchor variants (with/without the year, lightly reworded) -- 2-4 such variants are
  realistic before you decompose. But two *decomposed* sub-queries must never be paraphrases of each
  other ("NFCC find a counselor locator" and "NFCC find a counselor official locator" are one query,
  not two) -- if you're restating an angle to hit a count, emit fewer instead.
- NAME specific entities -- real engines search for specific *things*, not abstractions. Name real
  products/brands AND the authoritative sources a researcher would trust for THIS topic: regulators
  (FTC, IRS...), standards or professional bodies (ENERGY STAR, a medical society...), independent
  review publications appropriate to the *category* (consumer-goods reviewers for physical products;
  software-review platforms / B2B directories like G2 or Capterra for software; etc. -- match the
  reviewer to what's being bought, don't reach for a famous consumer-goods reviewer on a category it
  doesn't cover), or the official manufacturer/product page. Prefer entities a *neutral* researcher
  would trust. Do NOT guess the names of commercial service providers (lead-gen or vendor brands)
  unless the question itself names them -- engines search the category and its watchdogs, not
  individual vendors. When a question turns on a key NUMBER (an average rate, a standard threshold, an
  official statistic), name the body that publishes it (a central bank, a government statistical
  release, a standards body), not just a generic "average X" query -- engines go to the source of record.
- MATCH the source to the question type. Advice / "is it worth it" / "how does X work" / "is this a
  scam" questions lean almost entirely on neutral authorities (regulators, nonprofits, standards
  bodies) -- NOT commercial aggregators. Shopping / "best X" / product questions lean on independent
  review publications AND the official product/pricing pages. Don't put affiliate-style aggregators
  on regulatory or definitional questions.
- GO DIRECT to primary sources. For product, pricing, or spec questions, engines go straight to the
  official page -- both as plain text ("Wells Fargo Reflect Card 0% intro APR terms official") AND as
  site-scoped searches ("site:wellsfargo.com Reflect Card balance transfer fee"). For questions about
  specific buyable products, include several direct-to-source queries -- one per leading product --
  rather than only head-to-head comparisons.
- ANCHOR to recency where it matters. The current year is {year} -- use {year} (not an older year)
  when a date sharpens the search.
- COVER the sub-intents that genuinely matter FOR THIS QUESTION -- and only those (cost/pricing,
  specs, comparisons, reviews/quality, alternatives, etc.). Force nothing. Do NOT default to
  how-to / step-by-step sub-queries -- engines rarely decompose an informational or shopping
  question that way; add one only when the question is itself procedural ("how do I...", "how to
  set up..."). A forum/Reddit query is fine where real-user opinion genuinely matters (e.g. "is X
  worth it", product picks) -- just don't spray it onto every question.
- Match real specificity: these searches are often long and stacked with qualifiers, not
  two-word keywords.

Return ONLY valid JSON of this shape (no prose, no markdown fences):
{
  "fanout_plan": "<one sentence: the distinct sub-intents you see, and how many sub-queries that warrants>",
  "sub_queries": [
    {
      "sub_query": "<the search string an engine would issue>",
      "intent": "<one of: reformulation | named_entity | comparison | cost_price | specs_details | reviews_quality | how_to | alternatives | recency | other>",
      "priority": "<high | medium | low>"
    }
  ]
}
{persona_block}
USER QUERY:
{query}

===PERSONA-ADDON===
One more thing before you fan out: a specific person is doing this search. The same question
expands differently depending on who's asking -- let this persona's situation, constraints,
prior knowledge, and unstated worries decide WHICH sub-intents dominate and which specific
entities show up. For a sensitive situation (money, health, safety) this may legitimately
surface trust/legitimacy or eligibility questions; for a confident expert it may skew toward
specs and comparisons. Follow the person -- don't add these angles unless the persona pulls them.

Also add a "persona_driver" field to each sub-query naming the persona attribute that prompted it.

BUYER PERSONA:
{persona}
