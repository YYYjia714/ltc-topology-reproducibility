# Third-Party and Access Notices

## AMASS and Body Models

The study uses CMU, KIT and BMLmovi through AMASS. Raw motion parameter files,
extracted joint arrays, processed training windows, and SMPL/SMPL-H/SMPL-X body
model assets are deliberately excluded. Obtain them from their original providers
under the applicable terms; this repository grants no rights to those assets.

AMASS terms: https://amass.is.tue.mpg.de/license.html

## HisRepItself

Upstream: https://github.com/wei-mao-2019/HisRepItself

Reference commit: `0451c84491cf2b3697e373ce26dc75623ccaa89e`.

No upstream repository, body-model assets or pretrained weights are bundled.
The study's local `model/AttModel.py` difference from that commit is a trailing
newline only. Obtain dependencies directly and check their upstream terms.

## Controlled Method Adaptations

The authors' `code/src/reviewer_baselines.py` implements controlled common18
adaptations of MSR-GCN, ST-Transformer, siMLPe and HumanMAC. See the frozen plan
for architecture-family provenance. These are not redistributions of complete
third-party repositories and should not be described as exact reproductions of
published benchmark scores.

## Model Access

The authors elected to release code and analysis first, while keeping the 48
trained checkpoint files unpublished. Their checksums may be inspected in
`checkpoints_manifest.json`, but that inventory is not a download entitlement
or a public model license. The existing Zenodo record remains a draft.

## Scope of MIT

The existing MIT license applies to the authors' code. It does not relicense
third-party libraries, input datasets, body models, manuscript copyright or
checkpoint weights. Original frozen protocol records are historical evidence;
their earlier future-release wording is superseded by the current README.
