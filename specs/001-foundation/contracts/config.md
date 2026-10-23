# Contract: Config Files

All three files are YAML. They are validated on load, and errors name the entry and the problem.

## config/labels.yaml (committed)

```yaml
guideline_version: "0.1.0"
labels:
  - name: singing
    group: activity            # activity|category|actor|audio|expression|indicator|status
    prompts:                   # 3–5 phrasings
      - "a person singing on stage"
      - "a group of kids singing together"
      - "someone singing into a microphone"
    color: "#1f77b4"           # hex display color
```

Rules:

- `name` is unique.
- `group` must be in the allowed set.
- `prompts` must have 3–5 entries.
- `color` must match `#RRGGBB`.
- Every group needs at least one label.
- Expression labels are exactly `confident`, `upset`, `positive`, `neutral`, `negative`.
- No clinical, psychological, health, disability, demographic or mental-state labels (Constitution V). A banned-term list is checked in the tests.

Starter vocabulary (wording owned by the team):

| group | labels |
|---|---|
| activity | singing, dancing, acting, playing_instrument, speaking, rehearsing |
| category | performance, practice, workshop, discussion |
| actor | solo, small_group, ensemble, staff_led |
| audio | music, speech, applause, silence |
| indicator | participation, self_expression, original_work |
| status | uncertain, mixed, needs_review |

## config/pipeline.yaml (committed)

```yaml
window_s: 8
overlap_s: 2                   # must be < window_s
frames_per_window: 4           # ≥ 1
min_face_px: 48
label_threshold: 0.5
video_phases: [early, mid, showcase, unknown]
audio_sample_rate: 16000
moments:
  primary_group: activity
  secondary_groups: [category, actor, audio, expression]
  max_gap_windows: 1           # merge across gaps up to this many windows
  min_windows: 1               # drop moments shorter than this many windows
  needs_review_below: 0.6
  conflict_margin: 0.1
overrides:                     # only window_s, overlap_s, frames_per_window allowed
  by_kind:
    clip: { window_s: 4, overlap_s: 1 }
  by_video: {}                 # e.g. v03: { frames_per_window: 8 }
```

Override precedence is defaults < `by_kind` < `by_video`. The effective values must still satisfy `overlap_s < window_s`.

## config/videos.example.yaml (committed, mock only) / config/videos.local.yaml (gitignored, never read by assistants)

```yaml
videos:
  - video_id: mock01
    filename: mock/mock_rehearsal_01.mp4   # relative to DATA_DIR
    orientation: landscape
    kind: live
    phase: early
    cohort: mock-cohort-a
    split: dev
```

Rules:

- `video_id` is unique, opaque, and contains only `[a-z0-9]`.
- `filename` is relative, with no `..`.
- Enums must follow [tables.md](tables.md).
- `duration_s` and `fps` are probed from the file, not configured.

The loader reads `videos.local.yaml` when it exists and otherwise falls back to `videos.example.yaml`. Mock and test runs pass the example explicitly.
