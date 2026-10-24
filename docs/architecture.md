# Architecture: TPA Video Index

```mermaid
flowchart TD
    V[("Program videos<br/>v01–v12<br/>(Mac + HPC only)")] --> SEG["Segmenter<br/>8 s windows, 2 s overlap<br/>sampled frames + audio"]

    SEG --> SIG["SigLIP2<br/>visual label scores"]
    SEG --> WH["faster-whisper<br/>speech yes/no<br/>(text stays on HPC)"]
    SEG --> AST["AST<br/>audio cues"]
    SEG --> YF["YOLO11 people count<br/>+ YuNet faces → EmotiEffLib<br/>expression suggestion"]

    SIG --> TAB[("Per-segment tables<br/>segments · predictions<br/>(Parquet runs)")]
    WH --> TAB
    AST --> TAB
    YF --> TAB

    TAB --> RUL["Rules + learned model<br/>dev-calibrated thresholds,<br/>signal rules, logistic regression"]
    RUL --> MOM[("Moments<br/>label · time · confidence<br/>needs-review flag")]

    MOM --> APP["Review app (local, 127.0.0.1)<br/>Blind labeling · Review<br/>Timeline · Insights"]
    APP --> LOG[("Review log<br/>append-only, guideline-versioned")]

    LOG --> EVAL["Evaluate<br/>blind labels on eval videos,<br/>bootstrap CIs, kappa, ablation"]
    LOG --> INS["Insights<br/>youth-only indicators,<br/>early vs showcase, tagged"]
    MOM --> EVAL
    MOM --> INS
    INS --> APP

    LOG --> EXP["Export<br/>one record per moment<br/>JSON + CSV"]
    MOM --> EXP

    classDef hpc fill:#EDE6F7,stroke:#4100AD,color:#201C25
    classDef mac fill:#FFFFFF,stroke:#7851B9,color:#201C25
    classDef store fill:#F8F7F8,stroke:#6F717A,color:#201C25
    class SEG,SIG,WH,AST,YF hpc
    class RUL,APP,EVAL,INS,EXP mac
    class V,TAB,MOM,LOG store
```

## Legend

| Colour | Where it runs |
|---|---|
| Purple fill | **Heavy processing on UNF HPC** (one GPU per job): segmenting, SigLIP2, faster-whisper large-v3, AST, YOLO11m, YuNet + EmotiEffLib. The same stages also run on the Mac with smaller models (SigLIP2 base, faster-whisper small, YOLO11n), which the Mac benchmark measures. |
| White fill | **On the Mac** (or any machine TPA owns): rules and moments, the review app, evaluation, insights and the export. No GPU needed. |
| Grey fill | **Stored data.** |

## Where data lives

- **Videos**: in a data folder on the Mac and in UNF HPC private folders, outside the repo. Never in git or the cloud.
- **Per-segment tables and moments**: one immutable Parquet folder per run, in an outputs folder outside the repo. Runs are synced from HPC to the Mac as Parquet only. A config file lists which runs count as official.
- **Transcripts**: only on HPC, inside the transcript run folder. Never synced, shown or exported.
- **Review log**: a local database on the Mac. It is append-only, and every row carries reviewer, time and guideline version.
- **Results and exports**: evaluation and insights results, annotation exports and benchmark summaries go to the outputs folder on the Mac. Only aggregates are written, with no filenames, transcript text or per-person fields.
- **In the repo**: code, configs (labels, pipeline settings, theme), mock data tools and the mock-only Devpost sample. Never footage, filenames or real metadata.
