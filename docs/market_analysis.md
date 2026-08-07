# Regulatory and Competitive Landscape, 2022–2026

This document summarises the technical regulation, personnel, and
constructor changes across the seasons used in this project, and states
the modelling implications that follow from them. It is the basis for the
train / validation / prediction split adopted in `src/config.py`.

## Season split adopted

| Role | Seasons | Rationale |
|---|---|---|
| Training | 2022, 2023, 2024 | Three complete seasons under one stable regulation set |
| Validation | 2025 | Final season of the same regulation set; unseen at training time |
| Prediction | 2026 | First season of a new regulation set |

## 1. Technical regulations

**2022–2025 — "ground-effect" era.** The 2022 season introduced a full
technical reset: ground-effect underfloor aerodynamics returned for the
first time since 1982, replacing the previous generation of cars. This
same chassis philosophy, with incremental updates, carried through 2023,
2024, and 2025. The power unit formula (1.6L V6 turbo-hybrid, introduced
in 2014, roughly 80% ICE / 20% electric split) was unchanged across all
four seasons.

**2026 — new generation.** 2026 introduces the largest regulation change
since 2014: a new power unit formula (50/50 ICE/electric split, MGU-H
removed, maximum electric deployment raised from 120kW to 350kW,
100% sustainable fuel), together with a new chassis (smaller, lighter
cars) and active aerodynamics replacing DRS ("Overtake Mode"). Multiple
sources describe this as F1's most significant single-season regulation
change to date.

**Implication:** 2022–2025 form a single, internally comparable
regulation era — training on 2022–2024 and validating on 2025 tests
generalisation *within* a stable formula. 2026 is a distribution shift by
design, not an accident of the data. A model trained purely on 2022–2025
form should be expected to be less reliable on 2026, in the same way that
Mercedes' dominant 2014–2021 form did not predict their result after the
2022 reset (they fell to third in the constructors' standings). This
should be treated as a modelling risk to report on, not something to
paper over with more features.

## 2. Power unit manufacturers and supply changes

| Team | Pre-2026 supplier | 2026 supplier | Change |
|---|---|---|---|
| Red Bull Racing | Honda RBPT (rebadged Honda) | Red Bull Ford Powertrains | Red Bull now builds its own power unit for the first time |
| Racing Bulls (formerly AlphaTauri / RB) | Honda RBPT | Red Bull Ford Powertrains | Follows sister team |
| Aston Martin | Mercedes | Honda | New works supply deal |
| Alpine | Renault (works) | Mercedes (customer) | Ends decades of Renault works power |
| Sauber → Audi | Ferrari | Audi (works) | Becomes a full works team |
| Cadillac (new entrant) | — | Ferrari (customer) | New team, no prior data |
| Ferrari, Mercedes, Williams, McLaren, Haas | unchanged suppliers | unchanged suppliers | Continuity |

**Implication:** power unit changes are a second, independent source of
performance shift on top of the chassis reset, concentrated in specific
teams (Red Bull, Racing Bulls, Aston Martin, Alpine). Team-level
"historical pace" features should not be assumed to transfer for these
teams into 2026 to the same extent as for the teams with unchanged
suppliers.

## 3. Constructor identity changes

Several teams changed their official name across this window while
remaining the same operational entity:

| Seasons | Name in data | Becomes |
|---|---|---|
| 2022–2023 | Alfa Romeo (Sauber-run) | — |
| 2024–2025 | Sauber / Stake F1 Team Kick Sauber | — |
| 2026 | Audi | Full works rebrand, same Hinwil factory and core personnel |
| 2022–2023 | Scuderia AlphaTauri | — |
| 2024–2025 | RB / Visa Cash App RB | — |
| 2026 | Racing Bulls | Same Faenza factory, closer Red Bull integration |

**Cadillac** is a genuinely new entrant for 2026 (the 11th team, alongside
Audi) — the first brand-new constructor since Haas in 2016. It has no
history in this dataset at all.

**Implication:** raw constructor names cannot be joined across seasons
without normalisation, or every "Sauber → Audi" and "AlphaTauri → RB →
Racing Bulls" transition will be read by a model as a team disappearing
and a different team appearing from nowhere. `src/config.py` defines a
`CONSTRUCTOR_NAME_MAP` for this purpose. Cadillac has no mapping target —
it requires explicit cold-start handling (e.g. a rookie-team flag rather
than a rolling-form feature) rather than being silently dropped or
imputed with a false history.

## 4. Driver changes relevant to the modelled seasons

- **Lewis Hamilton** moved from Mercedes to Ferrari for 2025, ending a
  12-year Mercedes tenure. Mercedes replaced him with rookie Andrea Kimi
  Antonelli.
- **Max Verstappen** won the drivers' championship in 2022, 2023, and
  2024 (Red Bull). His four-year run ended in 2025, when **Lando Norris**
  (McLaren) won the title by two points over Verstappen, with teammate
  Oscar Piastri third — McLaren's first drivers' title since 1998/2008.
- **Sergio Pérez** left Red Bull at the end of 2024 (seat filled by Liam
  Lawson, then Yuki Tsunoda during 2025) and returns for 2026 with the
  new Cadillac team alongside **Valtteri Bottas**.
- **Constructors' championship:** Red Bull (2022, 2023) → McLaren (2024,
  2025).

**Implication:** driver-level form must be tracked by driver identity,
not by team, so that a driver's rolling form correctly follows them
across a mid-window team change (Hamilton, Pérez). Grid size also expands
from 20 cars (2022–2025) to 22 cars (2026, with Cadillac's two entries),
which affects any feature or metric defined relative to grid size (e.g.
"finished in the bottom 3").

## Summary of modelling implications

1. Train on 2022–2024, validate on 2025: a fair, within-era test of
   generalisation.
2. Report 2026 predictions with explicitly wider uncertainty bounds — it
   is a different regulation and supplier landscape by design, and the
   validation-set error rate is not a reliable estimate of 2026 error.
3. Normalise constructor names across seasons (`CONSTRUCTOR_NAME_MAP` in
   `src/config.py`) before computing any team-level historical features.
4. Track form by driver, not by team, so mid-window driver moves are
   handled correctly.
5. Flag Cadillac (and, to a lesser extent, any team with a changed power
   unit supplier for 2026) as cold-start cases rather than silently
   applying historical averages that don't apply to them.
