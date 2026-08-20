# Licensing

VibeSim is **source-available**, not OSI-approved open source. The source is
published and readable, and two community licenses cover a wide range of use at
no charge — but they are not open-source licenses, because they restrict the
field of use.

> This summary is provided for convenience only. In the event of a
> conflict, the applicable license text controls.

The controlling texts are:

- [`LICENSES/PolyForm-Noncommercial-1.0.0.txt`](LICENSES/PolyForm-Noncommercial-1.0.0.txt)
- [`LICENSES/PolyForm-Internal-Use-1.0.0.txt`](LICENSES/PolyForm-Internal-Use-1.0.0.txt)

Both are the unmodified official PolyForm texts. Nothing in this file, in
[`LICENSE`](LICENSE), in [`NOTICE`](NOTICE), or in
[`COMMERCIAL-LICENSING.md`](COMMERCIAL-LICENSING.md) adds to, removes from, or
reinterprets them.

## The licenses are alternatives

You choose the one that covers your use:

```
PolyForm Noncommercial 1.0.0
        OR
PolyForm Internal Use 1.0.0
```

You do not need to comply with both simultaneously. A university lab relies on
Noncommercial; a company's internal platform team relies on Internal Use;
neither has to care about the other.

## What each one is for

**PolyForm Noncommercial 1.0.0** permits any noncommercial purpose — research,
teaching, personal study, and noncommercial modification. It is the only one of
the two that permits distribution, and that distribution must itself be for a
noncommercial purpose and must carry the notices described in
[`NOTICE`](NOTICE).

**PolyForm Internal Use 1.0.0** permits use and modification for your own
company's internal business purposes, including commercial ones. It contains no
distribution grant at all: under this license you may not distribute the
software, modified or not.

Note what falls between them. A for-profit company may run and modify VibeSim
internally for free. The moment it hands the software — or a product built on
it — to anyone outside the company, neither community license covers that.

## Use-case table

| Use case | Community permission |
| --- | --- |
| Personal study | PolyForm Noncommercial |
| Academic research | PolyForm Noncommercial |
| University research | PolyForm Noncommercial |
| Educational use | PolyForm Noncommercial |
| Noncommercial research fork | PolyForm Noncommercial |
| Noncommercial redistribution | PolyForm Noncommercial, subject to its terms and notices |
| For-profit company using the software internally | PolyForm Internal Use |
| Company modifying the software for internal use | PolyForm Internal Use |
| Internal engineering / R&D | PolyForm Internal Use |
| Selling copies of the software | Commercial license required |
| Commercial redistribution | Commercial license required |
| OEM / white-label distribution | Commercial license required |
| Shipping it as part of a customer-facing commercial product | Commercial license required unless separately authorized |
| Offering the software's functionality as a commercial hosted/managed product | Commercial license required unless separately authorized |
| Proprietary commercial fork distributed to customers | Commercial license required |

Commercial organizations may use the software internally under the PolyForm
Internal Use License 1.0.0. External commercial uses not covered by either
community license require a separate commercial license — see
[`COMMERCIAL-LICENSING.md`](COMMERCIAL-LICENSING.md).

## Third-party software

Dependencies, vendored code, submodules, datasets, models, and examples may
carry separate licenses. **They are not relicensed merely by appearing in this
repository**, and the commercial license described here cannot grant rights in
them.

The two that matter most here:

- Files under `profiling/runners/attention/dsa_persistent_topk_native/upstream/`
  are byte-identical vLLM v0.23.0 source under **Apache-2.0**.
- `alignment/profiler/vllm` and `alignment/load_generator/req-frontend` are git
  submodules. Their contents are fetched from their own repositories and are not
  stored in this repository.

The inventory is in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## FAQ

**Can a university use it?**
Yes, when permitted under PolyForm Noncommercial 1.0.0.

**Can a researcher modify it and publish a noncommercial fork?**
Yes, subject to PolyForm Noncommercial 1.0.0 and its notice requirements.

**Can a for-profit company use it internally?**
Yes, when permitted under PolyForm Internal Use 1.0.0.

**Can a company modify its internal copy?**
Yes, when permitted under PolyForm Internal Use 1.0.0.

**Can a company redistribute its modified version to customers?**
Not under PolyForm Internal Use — that license grants no distribution right.
It would need another applicable license; commercial licensing is available
from the project owner.

**Can someone sell the project, or a proprietary fork of it?**
Neither community license should be read as granting that. Please go through
the commercial licensing process.

**Can someone run it as a paid SaaS or managed product for third parties?**
Do not treat that as automatically permitted community use. External commercial
hosted or managed offerings should go through the commercial licensing process.

**Do contributors lose copyright?**
No. Contributors retain copyright in their contributions and grant the rights
described in [`CLA.md`](CLA.md), which include commercial and proprietary
relicensing.

**Are dependencies relicensed?**
No. Third-party components remain under their own licenses.

**Is this OSI open source?**
No. It is source-available.

## Maintainer rule: preserving the ability to license commercially

This is a continuing maintenance rule, not a one-time migration step.

The project can only offer a commercial license covering the whole codebase if
it actually holds sufficient rights in every part of it. Therefore maintainers
must not merge a contribution unless the project has the right to relicense it
commercially:

```
No external contribution without CLA / relicensing rights.
No unknown copied code.
No silent incorporation of incompatible third-party code.
```

The enforcement details live in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Unresolved identifiers

Three placeholders are still unfilled. They are deliberately **three separate
tokens**, because they are three separate roles and they are not always the same
entity:

| Placeholder | Role | Appears in |
| --- | --- | --- |
| `<COPYRIGHT HOLDER>` | Who owns the copyright. A **factual** determination — who authored the code, and in what employment or institutional capacity. It is not a choice. | [`LICENSE`](LICENSE), [`NOTICE`](NOTICE) |
| `<PROJECT OWNER / LICENSOR>` | Who grants the licenses, receives CLA grants, and signs commercial agreements. Must either *be* the copyright holder or hold sufficient rights from them. | [`CLA.md`](CLA.md) |
| `<COMMERCIAL LICENSING CONTACT>` | An address for inquiries. Purely administrative. | [`COMMERCIAL-LICENSING.md`](COMMERCIAL-LICENSING.md) |

The second cannot be settled before the first: a licensor's authority is derived
from copyright ownership or from a grant by the owner. In the common cases they
collapse to one entity — but if the copyright belongs to an institution and is
licensed onward to another entity, they do not, and merging the tokens would
hide exactly that.

What this does and does not affect while unfilled:

- **The outbound licenses still operate.** A copyright grant does not depend on
  the holder being named in the file. What is missing is the `Required Notice:`
  line that downstream noncommercial redistributors would carry — a notice
  defect, not a failure of the grant.
- **No one gains commercial rights from the ambiguity.** Copyright defaults to
  all rights reserved; an unresolved name withholds permission, it does not
  create it. Neither community license grants external commercial distribution
  in the first place — Internal Use has no distribution clause at all, and
  Noncommercial's is limited to noncommercial purposes.
- **Inbound contributions are blocked**, and should stay blocked, until the
  licensor is named and acceptance is actually recorded. See
  [`CONTRIBUTING.md`](CONTRIBUTING.md).

Filling these in is a search-and-replace across the four files above.

## How the pieces stay separate

These are five different things and are deliberately kept in five different
documents. Do not merge them into one homemade license.

| Concern | Where it lives |
| --- | --- |
| Software copyright license | PolyForm Noncommercial / Internal Use, in `LICENSES/` |
| Commercial rights | a separately negotiated agreement; `COMMERCIAL-LICENSING.md` is only the route to one |
| Contributor rights | [`CLA.md`](CLA.md) |
| Third-party code | the original third-party licenses; [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) |
| Brand, name, logo | [`TRADEMARKS.md`](TRADEMARKS.md) |
