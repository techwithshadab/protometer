# Safety & compliance posture

BOTOX® (onabotulinumtoxinA) is a prescription medicine with serious risks, including a boxed
warning for the distant spread of toxin effect. A public chatbot answering questions about it is a
regulated-content surface, and this assistant is built to the standard that implies: **it informs,
it never advises, and it never lets a wrong or unsafe answer through.**

## The six controls

### 1. Grounded-only answering (anti-hallucination)
The model is instructed to answer **only** from the retrieved official-site content, and the egress
guard enforces it: a reply whose content does not overlap the retrieved context is **refused**, not
shown. On a medical topic a confident wrong answer is a safety failure, so the bot prefers "I can
only share what's on the official site, please ask your provider" over a plausible guess.

### 2. No medical advice (no dosing, diagnosis, or direction)
The system prompt forbids personal medical instruction. The egress guard independently **blocks**
replies that read as second-person directives ("you should take", "your dose", "stop taking", "I
recommend you…") and replaces them with a redirect to a healthcare provider. Informational
statements ("BOTOX® is used to treat chronic migraine") are fine; instructions to a specific person
are not.

### 3. Safety information is surfaced, never buried
Chunks containing safety content (boxed warning, Important Safety Information, side effects,
contraindications) are flagged at ingest and tracked through retrieval. When a reply draws on
safety-relevant material, the `safety` flag is set and the UI shows an Important Safety Information
note. The bot never downplays risk.

### 4. The visitor's PII is protected
The widget is public and will see personal data: names, emails, phone numbers, "my doctor is
Dr. X". Every such span is **tokenized at ingress** before it reaches retrieval, the model, or any
log. The model reasons over opaque tokens. The egress guard **blocks** any reply that contains the
visitor's PII or a raw token. Nothing personal is stored in the clear.

### 5. Respectful sourcing
The crawler honours `robots.txt` (which disallows the site's API, search, forms, and account
flows), stays on the botox.com host, rate-limits politely, and treats pages as **source to
summarize and cite**, never to reproduce verbatim at length. Citations link back to the official
pages so a user can read the authoritative source.

### 6. Emergency short-circuit (urgent-symptom detection)
BOTOX® carries a boxed warning for the **distant spread of toxin effect**, trouble swallowing,
speaking, or breathing that can appear hours to weeks after treatment. A visitor describing such
symptoms must not be routed through an informational Q&A flow. A deterministic classifier runs
**before retrieval and before any model call** (`_is_emergency` / `_EMERGENCY_RE` in
`orchestrator.py`, no LLM involved) and matches acute distress: trouble/inability to breathe,
swallow, speak, or move; choking, wheezing, shortness of breath; chest pain; severe allergic
reaction or swelling of the face/throat/tongue; loss of consciousness or vision; a paralysis/numbness
that is spreading or worsening; or an explicit call for emergency help.

On a match the turn **short-circuits to a fixed urgent-care message**, "This may be a medical
emergency… call your local emergency number (911 in the US) or get medical help right away… please
don't wait for an answer here", and no retrieval or generation happens. The classifier is
deliberately broad: a false positive costs one extra "seek help now" message; a false negative could
miss a real emergency, so it errs toward surfacing the warning. It is a safety net, **not** triage or
diagnosis, it never assesses severity, only routes obvious distress to emergency services.

## What the bot will not do

- Recommend or adjust a dose, or tell anyone to start/stop/change treatment.
- Diagnose, or interpret a user's symptoms.
- Answer from outside the official-site content (no open-web medical knowledge).
- Echo back or store the visitor's personal information.
- Present itself as a substitute for a healthcare provider.

## Escalation and refusal behaviour

Every unsafe or unanswerable path resolves to one of two calm, compliant responses:

- **Refusal** (can't answer from sources): *"I can only share general information from the official
  BOTOX® site, and I don't have that in my sources. For anything specific to your situation, please
  talk with your healthcare provider."*
- **Block** (reply failed a guard): *"I'm not able to help with that here. For medical guidance
  about BOTOX®, please consult your doctor. I can answer general questions using information from
  the official site."*

Both are returned with `refused`/`blocked` flags so the UI can present them distinctly, and neither
ever contains the model's rejected content or the user's PII.
