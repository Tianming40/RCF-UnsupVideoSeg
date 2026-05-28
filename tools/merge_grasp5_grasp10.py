#!/usr/bin/env python3
"""
Create CMC_grasp5_10_merged/ by combining grasp-5 and grasp-10 deinterlaced datasets.

Each sequence is renamed with a suffix (_g5 / _g10). Per-file symlinks are created
so filenames inside each subdir also carry the suffix, matching the path the
dataset loader constructs from train.txt.

Example for stem "96391832500300027230", suffix "_g5":
  JPEGImages/96391832500300027230_g5/
    96391832500300027230_g5.png   → CMC_grasp5_deinterlaced/JPEGImages/96391832500300027230/96391832500300027230.png
    96391832500300027230_g5_1.png → CMC_grasp5_deinterlaced/JPEGImages/96391832500300027230/96391832500300027230_1.png
  Flows_NewCT/96391832500300027230_g5/
    96391832500300027230_g5_1.npy → CMC_grasp5_deinterlaced/Flows_NewCT/96391832500300027230/96391832500300027230_1.npy
  BackwardFlows_NewCT/96391832500300027230_g5/
    96391832500300027230_g5_1.npy → ...

Usage:
  python tools/merge_grasp5_grasp10.py
"""

from pathlib import Path

GRASP5  = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp5_deinterlaced")
GRASP10 = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_deinterlaced")
DST     = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp5_10_merged")

SOURCES = [
    (GRASP5,  "_g5"),
    (GRASP10, "_g10"),
]


def link_file(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return False
    dst.symlink_to(src.resolve())
    return True


def process_source(src_root: Path, suffix: str, dst_root: Path):
    created = skipped = 0

    # ── JPEGImages ──────────────────────────────────────────────────────
    src_jpeg = src_root / "JPEGImages"
    dst_jpeg = dst_root / "JPEGImages"
    dst_jpeg.mkdir(parents=True, exist_ok=True)

    for seq_dir in sorted(src_jpeg.iterdir()):
        if not seq_dir.is_dir():
            continue
        stem = seq_dir.name
        new_stem = stem + suffix
        dst_seq = dst_jpeg / new_stem
        dst_seq.mkdir(exist_ok=True)

        # pre frame:  <STEM>.png  → <STEM><SFX>.png
        pre_src = seq_dir / f"{stem}.png"
        if pre_src.exists():
            c = link_file(pre_src, dst_seq / f"{new_stem}.png")
            created += c; skipped += (not c)

        # post frame: <STEM>_1.png → <STEM><SFX>_1.png
        post_src = seq_dir / f"{stem}_1.png"
        if post_src.exists():
            c = link_file(post_src, dst_seq / f"{new_stem}_1.png")
            created += c; skipped += (not c)

    # ── Flow directories ─────────────────────────────────────────────────
    for flow_dir in ["Flows_NewCT", "BackwardFlows_NewCT"]:
        src_flow = src_root / flow_dir
        dst_flow = dst_root / flow_dir
        dst_flow.mkdir(parents=True, exist_ok=True)

        if not src_flow.exists():
            print(f"  [warn] {src_flow} not found")
            continue

        for seq_dir in sorted(src_flow.iterdir()):
            if not seq_dir.is_dir():
                continue
            stem = seq_dir.name
            new_stem = stem + suffix
            dst_seq = dst_flow / new_stem
            dst_seq.mkdir(exist_ok=True)

            # flow file: <STEM>_1.npy → <STEM><SFX>_1.npy
            for npy in seq_dir.glob("*.npy"):
                # replace original stem prefix with new_stem
                new_name = new_stem + npy.name[len(stem):]
                c = link_file(npy, dst_seq / new_name)
                created += c; skipped += (not c)

    return created, skipped


def read_txt(path: Path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def remap_line(line: str, suffix: str) -> str:
    parts = line.split()
    stem = parts[0].rstrip("/").split("/")[-1]
    new_stem = stem + suffix
    return f"JPEGImages/{new_stem}/ {new_stem}.png {new_stem}_1.png"


def main():
    # Remove stale directory symlinks from previous run if any
    for flow_dir in ["JPEGImages", "Flows_NewCT", "BackwardFlows_NewCT"]:
        d = DST / flow_dir
        if d.exists():
            for item in d.iterdir():
                if item.is_symlink():  # old dir-level symlinks
                    item.unlink()

    DST.mkdir(parents=True, exist_ok=True)
    (DST / "ImageSets").mkdir(exist_ok=True)

    for src_root, suffix in SOURCES:
        print(f"\n{src_root.name} → suffix '{suffix}'")
        c, s = process_source(src_root, suffix, DST)
        print(f"  file symlinks created: {c}   already existed: {s}")

    for split in ["train.txt", "val.txt", "trainval.txt"]:
        combined = []
        for src_root, suffix in SOURCES:
            src_txt = src_root / "ImageSets" / split
            if not src_txt.exists():
                print(f"  [warn] {src_txt} not found")
                continue
            for line in read_txt(src_txt):
                combined.append(remap_line(line, suffix))

        dst_txt = DST / "ImageSets" / split
        with open(dst_txt, "w") as f:
            f.write("\n".join(combined) + "\n")
        print(f"\nImageSets/{split}: {len(combined)} entries")

    print(f"\nDone. Output: {DST}")
    print("Update your training config:")
    print(f"  data_path: {DST}")


if __name__ == "__main__":
    main()
