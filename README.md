# GhostStock

A retail analytics MVP that flags SKU-store pairs where the system's recorded
stock-on-hand is almost certainly **wrong** — the item shows "in stock" but
real sales have silently gone to zero for an extended streak, which usually
means the physical stock is lost, misplaced, or stolen. It's never
reordered, and the gap quietly compounds for weeks. This tool ranks the
top-20 most suspicious SKUs per store per day so a manager can go verify
just those, instead of walking the whole store.

Fully offline, zero paid APIs, zero cloud, zero external CDN at runtime —
the dashboard is hand-built HTML/CSS/SVG, so once the page loads it needs
no internet connection at all.

## What's inside

| File | Purpose |
|---|---|
| `generate_data.py` | Builds the synthetic dataset: 800 SKUs x 25 stores x 18 months of daily sales + system inventory, with promos/seasonality/weekday patterns, and **hidden** phantom stock-outs injected into ~3% of sku-store pairs (ground truth, never shown to the detector). |
| `load_real_data.py` | Loader + cleaner for a real retailer's CSV export (flexible column names: `sku_code`/`item_code`, `store`, `date`, `qty_sold`, `qty_on_hand`, ...). |
| `data_io.py` | Shared helpers for loading the synthetic dataset / ground truth. |
| `detector.py` | The detection engine: streak detection, each SKU's own velocity baseline, promo suppression, Poisson-implausibility scoring + an IsolationForest cross-check. Hard-capped at 20 alerts per store-day. |
| `evaluate.py` | Scores the detector against the hidden `ground_truth.csv`: precision@20, recall, and an estimated annual ₹ recovered, with arithmetic printed step by step. Also writes `data/eval_summary.json` so the dashboard loads instantly instead of re-running a multi-minute scan. |
| `server.py` | FastAPI backend — a thin JSON API over the detection engine, plus serves the dashboard. |
| `auth.py` | Admin-provisioned accounts: SQLite-backed users table, PBKDF2 password hashing, cookie sessions. No public signup — see "Access & accounts" below. |
| `static/` | The dashboard itself: `index.html` (app shell), `login.html`, `styles.css`, `app.js`. Vanilla JS, hand-built SVG charts, no build step, no npm, no CDN script tags. |

## Setup

```bash
pip install -r requirements.txt
```

## Run it

```bash
# 1. Generate the synthetic dataset (~20-40s, one-time)
python generate_data.py

# 2. Evaluate the detector against hidden ground truth — prints precision/recall
#    to the terminal AND writes data/eval_summary.json for the dashboard
#    (takes ~3-4 minutes; run once, before you start recording a demo)
python evaluate.py

# 3. Launch the dashboard
python server.py
```

Then open **http://localhost:8000** — you'll land on a login page. The very
first time `server.py` runs, it prints a generated admin login to the
console, e.g.:

```
First run — an admin account was created:
  Email:    admin@ghoststock.local
  Password: ru9JiBfr4zqB
```

Use that to sign in. No API keys, no cloud deployment, no email delivery
required.

## Access & accounts

**There is no public sign-up — on purpose.** This is an internal retail-ops
tool sitting on a mall's live inventory data, not a consumer product. The
correct model is: the mall has one admin (the GhostStock owner/IT contact),
and that admin creates a login for every store manager who needs access,
from the **Users** tab in the sidebar — email + a temporary password the
admin hands them directly (in person, over chat, however), optionally
**scoped to just the store(s) that manager runs**. A manager who logs in
only ever sees their assigned store(s) — the store dropdown, every API
call, and even a hand-crafted request for another store's data (403'd
server-side, not just hidden in the UI) are all restricted to that scope.
An admin can also edit an existing user's role/store scope later from the
same Users tab — access isn't fixed at creation time.

Everyone — including the admin — can replace an admin-assigned password
with one only they know, from the lock icon next to Log out in the
sidebar. That's the normal way to stop using a temporary password; an
admin's **Reset password** button on the Users tab is the separate,
"they're locked out, force-set something new" path.

How it's built, concretely:
- Accounts live in a local SQLite file (`data/ghoststock.db`, gitignored —
  never commit it, it holds password hashes). No external auth service.
- Passwords are hashed with PBKDF2-HMAC-SHA256 (200k iterations) and a
  random per-user salt — never stored in plain text.
- Sessions are an opaque token in an httponly cookie, valid 12 hours,
  tracked in memory — restarting the server logs everyone out. Fine for a
  single-process local deployment; swap `auth.SESSIONS` for a persisted
  store before running multiple worker processes.
- Pin the seeded admin's credentials instead of letting them be random by
  setting `GHOSTSTOCK_ADMIN_EMAIL` / `GHOSTSTOCK_ADMIN_PASSWORD` before the
  first run.
- The last remaining admin account can't be deleted, or demoted to
  manager, from the Users panel — there's always a way back in.
- Login is throttled: 5 failed attempts for an email locks it out for 60
  seconds (in-memory, resets on restart). A basic brute-force guard for a
  LAN-facing tool, not a substitute for a real rate limiter if this is
  ever exposed to the open internet.

### Using your own data (e.g. a real retailer's export)

In the dashboard's top-right toggle, switch to **"Upload real CSV"** and drop in
a CSV with (reasonably named) columns for SKU, store, date, quantity sold, and
quantity on hand. `load_real_data.py` handles common naming variants and basic
cleaning (missing values, inconsistent codes, duplicate rows, day-first date
ambiguity) automatically. Standalone:

```bash
python load_real_data.py path/to/your_export.csv
```

## How detection works (plain language)

For every SKU-store pair, as of a given date:

1. **Streak** — how many consecutive days has it sold **zero** units while
   the system shows stock **greater than zero**? (Vectorized pandas
   groupby, not a slow per-row loop — scores a whole store in well under a
   second.)
2. **Baseline** — what's this *specific* SKU's own normal daily sales pace
   (its own history, not a chain-wide average)? A fast mover only needs a
   few zero-sales days to look wrong; a naturally slow mover needs many. A
   SKU needs at least a few real selling days in its history before it's
   eligible at all — otherwise a SKU that's simply never been stocked looks
   identical to a genuine phantom.
3. **Poisson implausibility** — given that baseline rate, what's the chance
   of seeing this many zero-sales days by pure luck? `confidence ~= 1 -
   exp(-baseline_rate x streak_len)`. Tiny chance -> high confidence it's a
   phantom, not just a quiet SKU.
4. **Promo suppression** — if a promotion was running through much of that
   streak, the signal is muddier, so confidence is discounted.
5. **Stock magnitude** — a streak against "1 unit left" is less alarming
   (and less costly) than the same streak against "30 units left."
6. **IsolationForest cross-check** — an unsupervised anomaly pass over that
   store-day's whole SKU snapshot, blended in as a secondary signal so the
   ranking isn't purely one hand-tuned formula.

Only the top 20 highest-confidence alerts per store per day are ever shown —
a hard budget so the output stays usable by a human.

## Demo Script (for a 90-second recording)

1. **Show the data loading** — open a terminal, run `python generate_data.py`
   (or just point at the already-generated `data/` folder) and narrate: *"800
   SKUs, 25 stores, 18 months of daily sales and system inventory, with
   hidden phantom stock-outs seeded in for evaluation."*
2. **Launch the dashboard** — `python server.py`, open `localhost:8000`, sign
   in with the admin login printed in the terminal. Leave the toggle on
   **Synthetic demo**, pick a store, and show **today's top-20 ranked alert
   list** — point out a couple of plain-English reasons, e.g. *"14 zero-sales
   days while system shows 6 in stock, and no promo is active."*
3. **Click a flagged SKU row** and show the **sales-vs-stock chart** — the
   orange sales bars flatline to zero while the blue system-stock line stays
   up; hover to show the synced tooltip calling out the exact gap.
4. **Switch to the terminal** and run (or show the already-run output of)
   `python evaluate.py` — precision@20 and recall against the hidden ground
   truth, plus the step-by-step arithmetic.
5. **Back in the dashboard**, scroll to **"Estimated Annual ₹ Recovered"** and
   expand "Show the arithmetic" to show the same calculation live in the UI.

## Notes on the detection design

- Everything is scored *per store-day*, using only data up to and including
  the as-of date — no lookahead into the future, so the demo is honest about
  what a manager would actually know on that day.
- `MIN_STREAK_DAYS`, `MIN_SELLING_DAYS`, and `TOP_N` live at the top of
  `detector.py` if you want to tune aggressiveness.
- `ground_truth.csv` is **only** read by `evaluate.py` and the dashboard's
  rupee-recovered panel — the detector itself never sees it. That separation
  is what makes the precision/recall numbers meaningful rather than circular.
- The dashboard's "Recompute live" button on the rupee panel re-runs the same
  multi-minute scan `evaluate.py` does, from the browser, if you want to show
  it isn't a canned number — it's just slow, so it's off by default in favor
  of the precomputed `data/eval_summary.json`.
