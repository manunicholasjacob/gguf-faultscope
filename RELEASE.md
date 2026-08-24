# Release sequence

Five steps, in this order. The order is not cosmetic: prong 2 established that a release cut
before the Zenodo toggle is never archived, and six of the existing repositories got it wrong.

**Every step here is his.** Nothing in this file has been done.

## Before anything

```bash
python src/gguf_faultscope.py --selftest
python src/gguf_inject.py --selftest
python src/run_study.py --dry-run --n 5 --out /tmp/t.jsonl
```

All three pass on a clean clone with no model and no GPU. If any fails, stop.

## The five steps

**1. Create the repository, private, and push everything including the metadata files.**

`CITATION.cff` and `.zenodo.json` must be in the first push. Zenodo reads them at archive time,
and adding them after the release means the archived record carries defaults instead.

```bash
gh repo create manunicholasjacob/gguf-faultscope --private --source . --push
```

**2. Make it public.**

Zenodo cannot archive a private repository. This also starts the six-month public-history clock
that JOSS requires, which matters if a software paper is ever a target for this one.

**3. Turn the Zenodo toggle on, and confirm it took.**

Go to zenodo.org, GitHub tab, **Sync now**, wait, **Sync now again**. The list is cached and one
sync frequently does not show a repository created minutes earlier. Then flip the toggle for
`gguf-faultscope` by clicking the switch itself rather than the row; the row is not the control.

**Do not confirm this by looking at it. Confirm it with a command.** Flipping the toggle installs
a webhook on the GitHub side, so GitHub's own API answers the question without logging in to
Zenodo at all:

```bash
gh api repos/manunicholasjacob/gguf-faultscope/hooks \
  --jq '[.[] | select(.config.url | test("zenodo"))] | length'
```

**1 means the integration is live. 0 means it is not, whatever the web page appeared to show.**
Measured 24 August 2026, this repository returned **0** while nine of the ten already-archived
repositories returned 1, which is what the check is for. Run it across everything at once:

```bash
for r in $(gh repo list manunicholasjacob --limit 40 --json name,isPrivate \
             --jq '.[]|select(.isPrivate==false)|.name'); do
  printf "%-32s %s\n" "$r" \
    "$(gh api repos/manunicholasjacob/$r/hooks \
         --jq '[.[]|select(.config.url|test("zenodo"))]|length')"
done
```

This is the check that would have caught the six repositories in this campaign that cut a release
before the toggle and were never archived. It takes a second and it does not depend on reading a
cached page correctly.

**4. Only now, cut the release.**

```bash
git tag -a v0.1.0 -m "First release: exposure model, injection harness, and the P100 campaign"
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --notes-file RELEASE_NOTES.md
```

Zenodo archives on the release webhook. A release created before step 3 fires no webhook and is
never archived, and the fix is to cut another release rather than to wait.

**The tag is already pushed and it is inert.** `v0.1.0` points at the commit carrying both the
corrected `CITATION.cff` date and the README count reconciliation; a tag does nothing until a
release is created from it. So the only irreversible click in this whole file is the one at the
end of step 4, and step 3 must be verified with the command above before you make it.

**5. Put the concept DOI into `CITATION.cff` and the README, then push again.**

Zenodo issues two DOIs: a concept DOI that always resolves to the latest version, and a version
DOI for this release. **The concept DOI goes in `CITATION.cff` as `doi:`**, with both listed
under `identifiers`, matching the pattern in `ml-systems-lab`. Also replace
`date-released: "REPLACE-WITH-RELEASE-DATE"` with the real date, which is easy to miss.

## Checks after

- `CITATION.cff` parses. GitHub renders a "Cite this repository" box when it does, and shows
  nothing when it does not, which is the quickest test.
- The Zenodo record's license reads MIT, not "other". The `.zenodo.json` here uses the lowercase
  string `mit`, which is what Zenodo's vocabulary wants; `MIT` silently becomes "other".
- The record has no `isBasedOn` relation. It is `isSupplementTo` only.
- The archived files include `data/`, which is the part that makes it citable as a dataset
  rather than only as code.

## Then hand it to prong 3

They own the adoption tracker. Adding this repository on the day it goes public means the first
external issue or pull request is dated automatically rather than noticed late, and that event
is the named trigger for the critical-role criterion. It is a one-line change to their target
list.
