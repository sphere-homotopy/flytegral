# Third-party neural runtime

Flytegral's optional MaleCNS mode is built against a pinned source checkout of
[DOOMFLY](https://github.com/nftechie/doomfly):

- upstream repository: `nftechie/doomfly`
- pinned commit: `71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33`
- license: MIT, copyright 2026 nftechie and DOOMFLY contributors

Flytegral does not vendor or alter the MaleCNS wiring. `scripts/bootstrap-malecns.sh`
checks out that exact DOOMFLY revision locally under `.vendor/doomfly`, verifies the
upstream source-file hashes, and runs DOOMFLY's own `doom.connectome` and
`doom.prepare` pipeline.

The resulting graph is based on MaleCNS v1.0, released by the HHMI Janelia FlyEM
team, Cambridge Connectomics Group, Google Research and collaborators. The raw
connectome files retain their upstream data licensing and provenance. See the
pinned DOOMFLY `THIRD_PARTY.md`, `THIRD_PARTY_NOTICES.md`, dataset registry and
source lock for the authoritative notices and hashes.

The neural dynamics, retinal projection and Flytegral scalar readout are models,
not a reconstruction of measured membrane physiology or a claim that a biological
fly naturally computes definite integrals.

## Thinking buzz audio

The fly's thinking-state buzz uses `Bombus buzz.ogg` by Wikimedia Commons user
Mysid, recorded in Southern Finland. The copyright holder released the recording
into the public domain. Flytegral streams the original Ogg file from Wikimedia
Commons at runtime:

`https://upload.wikimedia.org/wikipedia/commons/c/ca/Bombus_buzz.ogg`

Source page: `https://commons.wikimedia.org/wiki/File:Bombus_buzz.ogg`
