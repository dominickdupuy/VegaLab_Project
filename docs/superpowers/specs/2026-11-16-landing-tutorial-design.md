# Landing page and tutorial: design

Date: 2026-11-16 · Status: approved in chat

## Problem

The home page opens on "What do you want to find out?" and assumes the visitor already
knows what the platform is for. A researcher arriving cold gets no statement of the
research question, the method or the finding, and nothing teaches the site's controls.
"How it works" exists but is reached only from the footer and the small VEGA LAB label.

## Audience

A quant or finance researcher first: someone who knows what the variance risk premium is
and wants to know whether this platform's answer can be trusted and how to use it.
Jargon is allowed; terms the glossary covers get an info tip.

## Routing and navigation

| Route | Before | After |
|---|---|---|
| `/` | the questions home | the new landing page |
| `/workbench/` | (none) | the old home, unchanged, including `?checks=1` and `?study=1` |
| `/tutorial/` | (none) | the tutorial, absorbing How it works |
| `/how-it-works/` | six-step page | a client redirect (`router.replace`) to `/tutorial/` (the site is a static export, so no server redirect) |

- The header's first tab reads **Workbench** (`/workbench/`). The brand name links to `/`.
  VEGA LAB and the footer link go to `/tutorial/`.
- The reviewer tour's two steps (`checks`, `study`) move from `/` to `/workbench/`, and
  their `done` checks accept the path with or without its trailing slash.
- How it works's `/#trust` link becomes `/workbench/#trust`.

## Landing page (`/`)

In order:

1. **Hero.** It asks "Is the variance risk premium a real edge, and can a learning agent
   find it?". One line on the method: a simulated SPX market with edges of known size,
   real SPX data, costs and a test fixed in advance. Two buttons: *Open the workbench*
   and *Take the tutorial*.
2. **Current finding.** The live `StudyFinding` (the same component the workbench
   uses). It renders nothing until the analysis is on the server; in that case a one-line
   note says where the result will appear.
3. **What you can study.** Four cards. Each card gives its question, one line of method,
   the page to open and the guided tour that answers it:
   - Is the premium real? → Real data, Sweep · tour `edge`
   - Can an agent capture it? → Synthetic alpha · tour `alpha`
   - What eats the edge? → Launch · tour `costs`
   - How big must a test be? → Power lab · tour `size`
4. **Why trust the numbers.** Three points: markets with known answers, paired seeds,
   and leak, replay and checks. Each links to its page and to the workbench trust band.
5. **Reproduce it.** `make web && make serve`, with links to REPRODUCE.md and
   PRE_REGISTRATION.md through `/api/docs/`.

The 3D surface stays on the workbench, so the landing page loads light.

## Tutorial (`/tutorial/`)

A table of contents (sticky on wide screens), then:

1. **Finding your way.** An annotated, simplified drawing of the header (brand, tabs,
   server status dot, copy link, theme), followed by short entries for the workflow
   strip, info tips, query-string settings and run keys.
2. **The research loop.** The six How it works steps, each with its pages and the
   matching guided tour.
3. **Your first ten minutes.** A numbered checklist that starts the `edge` tour and ends
   at reading the verdict, then points to the next tours.

## Code shape

- `web/app/page.tsx`: the landing page. The old body moves to `web/app/workbench/page.tsx`.
- `web/app/tutorial/page.tsx` with `tutorial.module.css`. `web/app/landing.module.css` for
  the landing page.
- Content lives in arrays at the top of each page, as How it works did.
- `web/lib/tours.ts` gains `tourHref(id)`: the first step of a tour with `&start=1`, the
  link `WorkflowCard` already builds.
- No new dependencies. Styling comes from `ui.module.css` and the global tokens, in both themes.

## Testing

- `npm run build` and `npm run typecheck` pass.
- `tests/api/test_server.py`'s page list gains `/workbench/` and `/tutorial/`.
- Manual check in the browser: both pages in light and dark themes, at desktop and phone
  widths. Every link resolves, `/how-it-works/` lands on `/tutorial/`, and the reviewer
  tour still completes its steps on `/workbench/`.
