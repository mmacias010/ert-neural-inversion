# EarthScope intern lightning talk — 5 minutes, 4 slides

**Revision 2**, rebuilt from practice-run feedback (Becky & Kelsey; Eli).
What changed and why is recorded at the bottom under *Feedback response*.

**Audience:** scientists, educators, software developers, communicators,
administrators, fellow interns. Curious, not experts. Almost nobody knows what
ERT is.

**One takeaway they should leave with:**
*Matching the measurements is not the same as getting the answer right.*

---

## The two rules that fix the two lowest scores

**1. Bullets are labels, not sentences.** Becky and Kelsey asked for bullet
points specifically — *"make sure the main points are highlighted (you want
people to focus on what you are saying, not reading the text)."* So: three
bullets per slide, three to six words each, no punctuation at the end. The test
is whether someone could read a bullet and get your point **without you**. If
they could, it is doing your job and it needs to be shorter. A bullet primes the
sentence you are about to say; it does not deliver it.

**2. You have time. Use it.** You were marked rushed *and* under five minutes,
which means you left 45–60 seconds unspent. This script is ~570 words — about
4 min 40 s at a genuinely slow pace. Do not speed up to fill the gaps; the
pauses are written in on purpose.

---

## Slide 1 — "Many pictures. Same measurements." *(~70 s · ends 1:10)*

**On the slide:** that phrase, large. One image — the survey line or the field
photo. (Reviewers loved the field photos; put one here.)

**Bullets:**
- Groundwater → drinking water, farms, ecosystems
- Electrodes measure how current flows
- Many pictures fit the same data

**Beats — say these in your own words:**

- Groundwater is drinking water, farms, ecosystems. Managing it means knowing
  where it sits underground.
- Electrodes in the ground, small current, measure how easily it flows.
  Wet ground conducts differently than dry rock. A computer works backwards to
  a picture of what's below.
- Name it once: **electrical resistivity tomography, ERT.**

**VERBATIM — the analogy. Learn this one cold:**

> Imagine you describe an animal to seven artists who've never seen one. Every
> drawing fits your description — but they draw seven different animals. Your
> description wasn't wrong. It just wasn't enough.
>
> *(pause)*
>
> That's my problem exactly. The measurements are the description. The pictures
> of the underground are the drawings.

**Beat to close:**

- So the computer has to assume something about what the ground should look
  like. Usually: that it changes smoothly. That assumption isn't in the data.
  It's a choice — and it decides the answer.

---

## Slide 2 — proof of the analogy *(~45 s · ends 1:55)*

**On the slide:** `inr_results/talk_nonuniqueness.png`. No heading needed — the
figure has its own.

**Bullets** (down the side, or beneath — they must not crowd the figure):
- Two methods, two different pictures
- Both predict the same   measurements
- Closer than our measurement error

**Beats — let the figure do the work, point as you talk:**

- Top row: two pictures of the same patch of ground, made by two different
  methods. Obviously not the same picture.
- Bottom: what each one predicts we'd measure. *(pause — let them see it)*
- One line. They agree with each other to within a percent and a half — and our
  measurements are only good to a percent and a half anyway.
- So the measurements cannot tell us which is right. Not with better math, not
  with a bigger computer. The information isn't there.

*This is the payoff of the artists analogy — the same claim, now shown. Do not
rush the pause after the bottom panel; the whole slide is that one beat.*

---

## Slide 3 — the seven methods *(~40 s · ends 2:35)*

**On the slide:** `network_results.png`, sharp-block row only, seven method
names readable. Heading: **"Seven artists. Same description."**

**Bullets — keep this slide the sparsest of the five:**
- 2 standard methods, 5 neural networks
- Same data, same physics, same start
- Only the assumption changes

*This is the slide Eli flagged for word count ("cut down the amount of words on
the second slide"). Three short bullets and the figure — nothing more. No error
numbers, no method descriptions.*

**Beats:**

- My project asked: what if a neural network makes that assumption instead?
- The physics doesn't change at all. Only how we describe the picture — instead
  of a value for every square in a grid, a network takes a position and returns
  a resistivity.
- Seven methods. Same measurements, same physics, same noise, same start.
- Truth on the left, attempts to the right. Some find both blocks. Some smear
  them together. One didn't finish.

*Point at panels while you talk. Pointing is slow, and slow is what you need.
This is a "look at this" slide, not an explaining slide — keep it moving.*

---

## Slide 4 — "Then I found the problem was me" *(~105 s · ends 4:20)*

**On the slide:** two numbers, very large, arrow between.
`50% gap` → `no gap`. Optionally the two Fourier panels below.

**Bullets — reveal one at a time if your software allows it:**
- "Fair" = same size, same time
- Some methods needed more
- 3 wrong conclusions, 1 shared setting

*One at a time matters here. All three at once and the audience reads your
punchline before you say it.*

**Beats:**

- First result was clear. The standard method won by about fifty percent.
- Before writing that down, I checked one thing: to be fair, I'd given every
  network the same size and the same amount of computing time.

**VERBATIM — the analogy returns. Slow, and pause where marked:**

> Then I looked at my own instructions. I'd given all seven artists the same ten
> minutes and the same brush — to be fair.
>
> *(pause)*
>
> But some of them needed more time. And some needed a finer brush.
>
> *(pause)*
>
> I hadn't been measuring their skill. I'd been measuring my instructions.

**Beats:**

- One network I'd written off as unable to solve the problem — it just needed
  more time than I'd allowed.
- The fifty percent gap? Once each method got the settings that suited it, that
  gap shrank to nothing.
- Three wrong conclusions. None of them came from the methods.

**VERBATIM — the line to land. Slowest sentence in the talk, then hold:**

> They came from a setting I'd chosen once and reused everywhere.
>
> *(pause — count two)*

**Personal significance — YOUR WORDS, one or two sentences.**
Both reviewers asked for this and it has to be yours. Prompts to answer:
*What did you believe about research before this summer that you don't now?*
*What was the moment you knew you'd found something real?*

Examples only — replace them, don't recite them:
- *"I came in thinking research was about getting the right answer. It turned
  out to be about earning the right to trust one."*
- *"Finding the flaw in my own experiment was the first time this felt like my
  work and not an assignment."*

---

## Slide 5 — the takeaway *(~25 s · ends 4:45)*

**On the slide:**
- **"Fitting the data isn't finding the answer."** — large
- **Names**, small: Hang Chen, Zhengyang Fang, Weiyu Guo, Chen Xiong ·
  University of Iowa
- **Funding**, small: RESESS · NSF National Geophysical Facility, operated by
  EarthScope Consortium · NSF award 2435260
- Logos, and a field photo behind if it doesn't fight the text

*Names and funding must be visible on the slide, not just spoken — Becky and
Kelsey asked for this specifically.*

**Beats:**

- Callback to slide 2 — two pictures fit those measurements equally well, and
  one of them was far more wrong. That's the whole thing.
- On real field data, matching the measurements is the only thing you can check.
- Anyone comparing methods holds some settings fixed to be fair. Those shared
  settings can quietly manufacture the result.
- Next: real field data from Iowa and Colorado.

**VERBATIM — say the names clearly, do not rush the ending:**

> Thank you to Hang Chen, Zhengyang Fang, Weiyu Guo, and Chen Xiong at the
> University of Iowa, and to RESESS, the NSF National Geophysical Facility, and
> EarthScope for making this summer possible.

---

## Pacing checkpoints — glance at the clock

Matched to the built deck (7 slides, title and acknowledgements included).

| Slide | Ends at | Length |
|---|---|---|
| 1 — Title | 0:05 | say your name and the title, then move |
| 2 — Background and Motivation | 1:10 | 65 s · the analogy lives here |
| 3 — Same data. Different Answers | 1:55 | 45 s · one long pause |
| 4 — Seven artists. Same description | 2:30 | 35 s · point, don't explain |
| 5 — Results and Observations | 4:05 | **95 s · this is the talk** |
| 6 — Personal Significance & Future Work | 4:30 | 25 s |
| 7 — Acknowledgements | 4:45 | 15 s · say the names slowly |

**The checkpoint that matters is 2:30.** If you're starting slide 5 before 2:10,
you are speeding — stop, breathe, and deliver the "same ten minutes and the same
brush" line at half the pace that feels natural. It will not feel too slow to
the audience. It never does.

**Slide 5 gets more than a third of the talk.** Everything before it is setup and
everything after is landing. If a timed run comes in long, cut from slide 4 —
the figure carries it with almost no narration.

**Under 4:15 total means you rushed again**, even if it felt fine. Both reviewers
marked you rushed *and* short. Aim for 4:30–4:50.

---

## Delivery notes

**Only four passages are verbatim** — the analogy, its return, the landing line,
and the acknowledgments. Everything else is beats. Say them in your own words,
differently each rehearsal. If you find yourself reciting the beats word for
word, rewrite them shorter.

**Slide 4 is the talk.** Slides 1–3 are setup; slide 5 is the landing.

**Numbers on slide 2, if anyone asks afterwards:** the two models differ from
each other by 1.34 in log-RMS; their predicted measurements differ by 1.46%
against 1.5% noise; both sit at chi-squared 0.99 and 1.00; model errors are
0.731 (SIREN) and 1.221 (Deep Image Prior). None of that goes in the talk.

**Words to avoid entirely:** regularization, hyperparameter, capacity,
chi-squared, log-RMSE, spectral bias, inversion (say "working backwards from the
measurements"), architecture (say "network design").

**Acronyms:** ERT is the only one you need, defined on slide 1.
**IRIS** — reviewers flagged that it appeared undefined. IRIS merged with
UNAVCO in 2023 to form EarthScope Consortium, so the simplest fix is to cut it
and say **EarthScope**. If a funding line genuinely requires IRIS, spell out
*Incorporated Research Institutions for Seismology* on first use.

**Numbers:** four in the whole talk — seven methods, fifty percent, three wrong
conclusions, two field sites.

**Title:** the rubric says *TBD* on both copies. Lock it before the next run:

> ## Fitting the Data Isn't Finding the Answer
> ### Benchmarking Deep-Learning Frameworks for Electrical Resistivity Tomography Inversion

---

## Practice checklist

- [ ] Title on slide 1 — it was TBD on both rubrics
- [ ] Three bullets max per slide, 3–6 words each, no end punctuation
- [ ] No bullet is a complete sentence
- [ ] Slide 4's bullets revealed one at a time
- [ ] Names and funding **on** slide 4, not only spoken
- [ ] Timed run landing 4:30–4:50 — under 4:15 means you rushed again
- [ ] Two rehearsals from beats only, notes face down
- [ ] The four verbatim passages memorized
- [ ] Personal-significance sentence written in your own words
- [ ] IRIS cut or spelled out
- [ ] Pause held after "a setting I'd chosen once and reused everywhere"

---

## Feedback response

| Feedback | Where it's addressed |
|---|---|
| Pacing — rushed, had time to spare (both) | Checkpoint table; ~570 words; written-in pauses |
| Too much text, slide 2 especially (both) | Ten-word rule; slide 2 stripped to figure + 5-word heading |
| Make an analogy (Eli) | Seven artists, slide 1 — and it returns on slide 3 |
| Don't read off a script (peers) | Script → beats; only 4 verbatim passages |
| Define / spell out IRIS | Delivery notes — cut it, or spell it out |
| Why the project mattered to you | Slide 3 close, your words, prompts given |
| Names/funding on final slide | Slide 4 slide spec |
| Field photos are great | Kept — slide 1 image |
| Length scored low by both | Checkpoint table; target 4:30–4:50, not "under 5" |
