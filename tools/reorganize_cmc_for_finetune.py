
import os
import random
from pathlib import Path

SRC_JPEG = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_medical_format_deinterlaced/JPEGImages/cmc_sequence")
DST      = Path("/media/mitiadmin/Micron_7450_1/tianming/dataset/CMC_grasp10_finetune_deinterlaced")

VAL_RATIO = 0.2
SEED      = 42


def build_split_line(base_id: str) -> str:
    pre  = f"{base_id}.png"
    post = f"{base_id}_1.png"
    return f"JPEGImages/{base_id}/ {pre} {post}"


def main():
    all_files = sorted(SRC_JPEG.glob("*.png"))
    pre_frames = [f for f in all_files if not f.stem.endswith("_1")]
    print(f"Found {len(pre_frames)} pairs in source")

    # Verify every pre frame has a matching post frame
    valid_pairs = []
    for pre in pre_frames:
        post = SRC_JPEG / f"{pre.stem}_1.png"
        if post.exists():
            valid_pairs.append(pre.stem)
        else:
            print(f"  [warn] missing post frame for {pre.stem}, skipped")
    print(f"Valid pairs: {len(valid_pairs)}")

    # Train / val split
    random.seed(SEED)
    shuffled = valid_pairs.copy()
    random.shuffle(shuffled)
    n_val       = max(1, int(len(shuffled) * VAL_RATIO))
    val_ids     = sorted(shuffled[:n_val])
    train_ids   = sorted(shuffled[n_val:])
    print(f"  train: {len(train_ids)}   val: {len(val_ids)}")

    # Create directory skeleton
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "ImageSets").mkdir(exist_ok=True)
    (DST / "Flows_NewCT").mkdir(exist_ok=True)
    (DST / "BackwardFlows_NewCT").mkdir(exist_ok=True)

    # Create per-pair JPEGImages subdirs with symlinks
    for base_id in valid_pairs:
        seq_dir = DST / "JPEGImages" / base_id
        seq_dir.mkdir(parents=True, exist_ok=True)

        for suffix in ("", "_1"):
            fname = f"{base_id}{suffix}.png"
            src   = SRC_JPEG / fname
            dst   = seq_dir / fname
            if not dst.exists():
                dst.symlink_to(src.resolve())

    # Write ImageSets txt files
    def write_txt(path: Path, ids: list):
        with open(path, "w") as f:
            f.write("\n".join(build_split_line(i) for i in ids))

    write_txt(DST / "ImageSets" / "train.txt",    train_ids)
    write_txt(DST / "ImageSets" / "val.txt",      val_ids)
    write_txt(DST / "ImageSets" / "trainval.txt", sorted(valid_pairs))

    print(f"\nDone. Output: {DST}")
    print("Next step: run RAFT/generate_flows_cmc.py to fill Flows_NewCT/")


if __name__ == "__main__":
    main()
