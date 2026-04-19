---
name: notebooklm-instructions
description: Generate a NotebookLM "Customize" prompt for a CHEME-5820 lecture. Reads the lecture notebook(s) in the target folder, fills in a 5-block template (audience, intuition, math highlights, tradeoffs, tone), and writes the result to `notebooklm-instructions.md` in the root of that lecture folder. Invoke when the user asks to "write NotebookLM instructions", "generate the NotebookLM prompt", or similar for a specific lecture.
---

# NotebookLM Instructions Skill

Generates the prompt the course instructor pastes into NotebookLM's
"Customize" box when producing audio overviews of CHEME-5820 lectures.

## How to invoke

The user will name a lecture, typically by its ID (e.g. `L14a`,
`L13c`) or by pointing at a folder. Resolve that to a lecture folder
under `lectures/week-*/<lecture-id>/`.

If the user does not specify, ask which lecture before proceeding.
Do not guess from the current working directory — the user may have
multiple lectures in flight.

## What to do

1. **Locate the lecture folder.** Confirm `CHEME-5820-<id>-Lecture-*.ipynb`
   exists there. If there is also a companion example notebook
   (`CHEME-5820-<id>-Example-*.ipynb`), read it too; otherwise just
   the lecture.

2. **Read the notebook(s).** Extract the genuine content: what problem
   the lecture solves, the central mathematical objects, the key
   derivation steps, and any tradeoff discussion already present in
   the text. Do not fabricate content that is not in the notebook.

3. **Fill in the 5-block template** (see below) using what you read.
   Keep the total prompt at roughly 500 characters of actual
   instruction content. NotebookLM ignores prompts much longer than
   that.

4. **Write the result** to `<lecture-folder>/notebooklm-instructions.md`
   using the output format specified below. Overwrite if the file
   already exists — the skill is idempotent.

5. **Report to the user.** One sentence saying where the file landed,
   and whether a companion example notebook was incorporated.

## The 5-block template

Each generated prompt has these five sections, in order. Compose them
into a single prose + bulleted block inside a fenced code block in the
output file (so the user can copy it directly into NotebookLM).

1. **Audience line.** One sentence naming student background and what
   they do not know yet. NotebookLM calibrates depth off this. For
   CHEME-5820, the default audience is "advanced undergrad / early
   graduate students in chemical engineering who know linear algebra
   and ODEs" — adjust the trailing "but are new to X" clause to the
   lecture topic.

2. **Opening intuition paragraph.** Two to three sentences. Frame
   *why the technique exists* and *what problem it solves*, in plain
   language. No formulas. Seed one analogy if you can find a natural
   one in the notebook text.

3. **Math highlights as a bulleted checklist.** Three to five items.
   Each item names a mathematical object *and why it matters*, not
   the full derivation. Describe formulas in words; NotebookLM reads
   raw LaTeX awkwardly. Prefix this block with the phrase "without
   dwelling on derivations" so the hosts do not try to walk through
   proofs out loud.

4. **Tradeoffs section with explicit upside / downside bullets.**
   This is the single biggest lever for a balanced episode. Without
   it the hosts default to cheerleading. Pull genuine downsides from
   the notebook (assumptions, frozen parameters, restricted settings,
   known failure modes). If the notebook has no downside discussion,
   derive plausible ones from the method's structure rather than
   leaving this section generic.

5. **Tone instruction.** One sentence. "Keep the tone conversational,
   not lecture-y. Prefer analogies over formulas read aloud."
   NotebookLM weights the tail of the prompt heavily, so the tone
   line goes last.

## Output file format

Write `notebooklm-instructions.md` in the lecture folder with this
structure:

```markdown
# NotebookLM Audio Overview Prompt: <lecture id> — <short title>

Paste the block below into NotebookLM's "Customize" box when
generating the audio overview for this lecture.

\`\`\`
<the filled-in 5-block prompt>
\`\`\`

## What was emphasized

- <one-line note on the opening intuition choice>
- <one-line note on which math highlights were included and why>
- <one-line note on the downside bullets, especially any that
  required inference beyond what the notebook states>
```

The "What was emphasized" section is short — it exists so the
instructor can eyeball what you picked without re-reading the
notebook. Three to five bullets total, one line each.

## Guardrails

- **Do not invent content.** Every math highlight and downside must
  be traceable to the notebook. If you are reaching, say so in the
  "What was emphasized" section.
- **No em dashes** (course-wide style preference). Use commas,
  parentheses, or sentence breaks.
- **Do not commit the generated file.** Leave that to the user.
- **Do not regenerate notebook markdown exports** as a side effect.
  This skill only writes the instructions file.
