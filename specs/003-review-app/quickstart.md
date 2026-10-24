# Quickstart: Review App

## On mock data (no footage)

```bash
uv run tpa-index mock --seed 0 --clips    # mock tables plus synthetic videos in DATA_DIR/mock/
# list the mock run as official (config/official_runs.yaml): predict: [mock-seed0]
uv run tpa-index app                      # open http://127.0.0.1:7860
```

## On the TPA Mac (staff)

1. Make sure `config/official_runs.yaml` lists the official runs (`tpa-index runs` helps).
2. Run `uv run tpa-index app` and open the printed `http://127.0.0.1:…` address. It works only on this machine.
3. Enter your first name or role. Never a participant's name.
4. **Blind labeling, main sample.**
   - `1`–`9` picks the primary activity, the letter keys toggle the other labels (shown next to each), and `U` marks uncertain/mixed.
   - `Enter` saves and moves to the next segment, `Backspace` goes back, `Space` replays.
   - Progress is saved on every save. Reopen the app any time to continue.
5. **Second labeler:** choose "Overlap set" (60 segments).
6. **Top-up:** appears after official predict runs cover the eval videos. It's still blind.
7. **Review:** eval videos appear once the main sample and the overlap set are done.
   - For each AI suggestion, choose accept, edit, reject or defer.
   - Add labels or observations as needed. Notes: no names, no judgments about a person.

Clips are cut into a temporary folder and deleted when you close the app. Nothing leaves the Mac.
