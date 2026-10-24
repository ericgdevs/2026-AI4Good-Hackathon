# Contract: Config Changes (003)

## config/pipeline.yaml: new `app:` section

```yaml
app:
  port: 7860                 # 127.0.0.1 only; next free port if taken
  seed: 2026                 # blind sample seed; changing it never reshuffles an existing queue
  sample_size: 300           # main blind sample (all eval segments if fewer)
  overlap_size: 60           # first N of the main sample, for the second labeler
  rare_min_positives: 10     # a label is rare with fewer blind positives than this in the main sample
  topup_per_label: 20        # top-up cap per rare label
  padding_s: 3               # review playback context before and after a moment
  clip_height: 480           # clip resolution for playback (speed over detail)
```

Validation:
- integers ≥ 1;
- `overlap_size ≤ sample_size`;
- `0 ≤ padding_s ≤ 30`;
- `clip_height` between 240 and 1080;
- `port` between 1024 and 65535.

There is deliberately no key for host, sharing or analytics: those are fixed in code.

## config/labels.yaml

```yaml
guideline_version: "0.3.0"
# Adding an observation type (or any label) requires bumping guideline_version.
observation_types:
  - {name: staging_issue,    description: "lighting, set or staging problem"}
  - {name: audio_problem,    description: "sound missing, clipped or noisy"}
  - {name: camera_problem,   description: "framing, focus or shake makes the moment hard to see"}
  - {name: off_task,         description: "activity unrelated to the session plan"}
  - {name: worth_showcasing, description: "a strong program moment for funders (no individual judgment)"}
  - {name: clip_unavailable, description: "footage could not be played (logged by Skip)"}
```

Validation:
- names are unique, `^[a-z][a-z0-9_]*$`, and different from every label name;
- each description is a non-empty string;
- `clip_unavailable` must exist.
