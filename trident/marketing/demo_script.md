# Trident Launch Demo Video Script

**Target Length:** 3 Minutes  
**Goal:** Show, don't just tell. From 'Hello World' to 'Production Invoice API' in real-time.

---

### Scene 1: The Problem (0:00 - 0:30)

**Visual:** Screen recording of a complex Python project structure.
**Voiceover:** "This is how we build AI pipelines today. Ten different libraries, massive Dockerfiles, slow imports, and fragile integration code just to read a PDF."
**Visual:** Show a terminal running a Python script. It hangs, then crashes with a CUDA error.
**Voiceover:** "It's slow, expensive, and painful to debug."

### Scene 2: The Trident Differences (0:30 - 1:00)

**Visual:** Clean VS Code window. Typing `pipeline InvoiceProcessor:`
**Voiceover:** "Meet Trident. The first programming language where AI is a primitive."
**Visual:** Typing `ocr.extract(img)` and `@intent("Extract vendor")`.
**Voiceover:** "We treat OCR and NLP just like strings and integers. No imports. No boilerplate. Just your intent."

### Scene 3: The 'Hello World' (1:00 - 1:45)

**Visual:** Running `trident run invoice.tri`.
**Voiceover:** "Let's run it against a real invoice."
**Action:** The terminal instantly shows the structured JSON output: Vendor: "AWS", Total: $450.00.
**Voiceover:** "That's it. Under the hood, Trident compiled this to JAX and ran it on a TPU simulator. It took 0.4 seconds."

### Scene 4: Type Safety & Validation (1:45 - 2:15)

**Visual:** Adding a `struct Invoice` with types.
**Voiceover:** "But it's not just a script. Trident is strictly typed."
**Action:** Show the IDE autocomplete the fields. Add a validation logic `if total > 1000`.
**Voiceover:** "You get compiler safety for your data shapes. If your model output doesn't match your schema, Trident catches it."

### Scene 5: The Hub & Deploy (2:15 - 3:00)

**Visual:** Running `trident hub publish`.
**Voiceover:** "When you're done, publish it to the Trident Hub."
**Action:** Show the "Successfully published" message.
**Visual:** Switch to browser showing `hub.trident.dev` with the new package.
**Voiceover:** "Now anyone on your team can install it with one command. Trident: Stop writing glue code. Start building intelligence. Star us on GitHub."

---

**End Card:**
[ Trident Logo ]
`pip install trident-lang`
github.com/trident-lang/trident
