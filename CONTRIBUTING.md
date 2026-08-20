# Contributing

Thanks for your interest in VibeSim.

Working conventions — environment, test tiers, formatting radius, the skill tree
— live in [`CLAUDE.md`](CLAUDE.md), [`AGENTS.md`](AGENTS.md), and
[`doc/README.md`](doc/README.md). This file covers the licensing side of
contributing: what you grant, what you must disclose, and what a maintainer has
to check before merging.

## Contributor License Agreement

VibeSim is source-available under two community licenses and is separately
available under commercial licenses (see [`LICENSING.md`](LICENSING.md)). For
that to hold together, the project must have the right to license *the whole
codebase*, including contributions, under all of those terms.

So external contributions require the CLA in [`CLA.md`](CLA.md).

> By contributing, you retain ownership of your contribution but grant
> the rights described in CLA.md.
>
> External pull requests must not be merged until CLA acceptance has
> been recorded.

You keep your copyright. What the project receives is a broad license —
including the right to relicense your contribution commercially and
proprietarily. If that is not acceptable to you, please open an issue to discuss
before writing code, rather than sending a PR that cannot be merged.

### CLA status — read this before accepting outside code

[`CLA.md`](CLA.md) is currently marked **DRAFT**: the project owner / licensor is
not yet filled in, and no signature-recording mechanism is configured.

There is **no CLA bot and no automated verification** in this repository. The
checkbox in the pull request template is an acknowledgement by the author, not a
signature, and not a legal record. A ticked box must not be treated as CLA
acceptance.

Until maintainers (a) complete the owner identity in `CLA.md` and (b) configure
an actual acceptance-recording mechanism — a CLA assistant, a signed-agreement
file, or an equivalent auditable record — **do not merge contributions from
outside the project owner.**

### Scope in time

> The CLA applies to contributions submitted after the CLA process is adopted.
> It does not retroactively change ownership or licensing rights for earlier
> contributions.

Adopting a CLA now grants nothing about the past. If earlier external
contributions turn out to need relicensing permission, that has to be obtained
from those contributors individually, through a separate consent process.

For the record, the audit run when this licensing structure was adopted found
one non-owner author in the history: a single mechanical `cargo fmt` commit
(`28ff93c`) by an internal collaborator, confirmed by the project owner as
internal. No other external authorship was found across 481 commits.

## Third-party material

Do not casually copy code from other projects into this repository. "It was on
GitHub" is not a license.

If your contribution includes or adapts third-party material, you must disclose
it in the pull request:

- what the material is;
- where it came from — project, URL, version or commit;
- its license;
- any restriction that travels with it.

Permissively licensed material may still carry notice requirements. Those
notices must be preserved and the component added to
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

### Flag for maintainer review

Code copied or adapted from software under any of the following needs maintainer
review **before** it is merged:

- GPL
- LGPL
- AGPL
- SSPL
- BSL
- other source-available licenses
- licenses with unusual redistribution restrictions
- unknown or custom licenses

This is not a claim that these licenses are unusable. It is that their
obligations can interact badly with this project's dual community licensing and
its ability to offer commercial licenses — so a human decides case by case,
rather than discovering the problem after the fact.

## Maintainer checklist

Before merging anything that is not your own work:

- [ ] CLA acceptance is **recorded**, not merely claimed in a checkbox.
- [ ] Third-party material is disclosed, licensed compatibly, and listed in
      `THIRD_PARTY_NOTICES.md` with its notices intact.
- [ ] No third-party copyright or license header has been removed or altered.
- [ ] The project would still be able to relicense the merged result
      commercially.

If the last item is not true, do not merge. This is a continuing rule, not a
one-time migration step:

```
No external contribution without CLA / relicensing rights.
No unknown copied code.
No silent incorporation of incompatible third-party code.
```
