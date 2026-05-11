import os, sys, argparse, shutil, subprocess, numpy as np
from pathlib import Path
from PIL import Image

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",      required=True)
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--stain_san_dir",  required=True)
    p.add_argument("--work_dir",       required=True)
    p.add_argument("--n_tiles",        type=int, default=300)
    p.add_argument("--npz_key",        default="arr_0")
    p.add_argument("--n_jobs",         type=int, default=8)
    p.add_argument("--extractor_type", default="svd")
    p.add_argument("--keep_work_dir",  action="store_true")
    return p.parse_args()

def to_uint8(tile):
    if not isinstance(tile, np.ndarray):
        tile = np.array(tile.tolist())
    if tile.dtype == object:
        tile = np.array(tile.tolist(), dtype=np.uint8)
    elif tile.dtype != np.uint8:
        tile = tile.astype(np.uint8)
    return tile

def run(cmd, cwd=None):
    print("  $ " + " ".join(cmd)); sys.stdout.flush()
    ret = subprocess.run(cmd, cwd=cwd)
    if ret.returncode != 0:
        print("[ERROR] exit code " + str(ret.returncode), file=sys.stderr)
        sys.exit(ret.returncode)

def main():
    args = parse_args()
    png_in  = os.path.join(args.work_dir, "png_input")
    png_out = os.path.join(args.work_dir, "png_output")
    for d in [png_in, png_out, args.output_dir]:
        os.makedirs(d, exist_ok=True)

    npz_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".npz")])
    print("[info] trovati " + str(len(npz_files)) + " NPZ"); sys.stdout.flush()

    print("\n[Step 1] Estrazione PNG..."); sys.stdout.flush()
    npz_map = {}
    for npz_fn in npz_files:
        stem = Path(npz_fn).stem
        arr  = np.load(os.path.join(args.input_dir, npz_fn), allow_pickle=True)[args.npz_key]
        N    = min(arr.shape[0], args.n_tiles)
        infos = []
        for i in range(N):
            fn_out = stem + "_" + str(i).zfill(4) + ".png"
            fp_out = os.path.join(png_in, fn_out)
            if not os.path.exists(fp_out):
                Image.fromarray(to_uint8(arr[i, 0]), mode="RGB").save(fp_out)
            infos.append((i, arr[i, 1], fn_out))
        npz_map[stem] = infos
        print("  " + npz_fn + ": " + str(N) + " tile"); sys.stdout.flush()

    print("\n[Step 2] san_images (normalizzazione)..."); sys.stdout.flush()
    run([
        "python", os.path.join(args.stain_san_dir, "scripts", "san_images.py"),
        "--train-input-dir",  png_in,
        "--train-output-dir", png_out,
        "--test-input-dir",   png_in,
        "--test-output-dir",  png_out,
        "--extractor-type",   args.extractor_type,
        "--n-jobs",           str(args.n_jobs),
        "--case",             "gaussian_one_zero_uniform_train_mixup",
        "--test-only",
    ], cwd=args.stain_san_dir)

    print("\n[Step 3] Ricostruzione NPZ..."); sys.stdout.flush()
    for npz_fn in npz_files:
        stem  = Path(npz_fn).stem
        infos = npz_map[stem]
        new_arr = np.empty((len(infos), 2), dtype=object)
        missing = 0
        for row, (_, v1, fn_png) in enumerate(infos):
            fp = os.path.join(png_out, fn_png)
            if not os.path.isfile(fp):
                fp = os.path.join(png_in, fn_png)
                missing += 1
            new_arr[row, 0] = np.array(Image.open(fp).convert("RGB"))
            new_arr[row, 1] = v1
        np.savez(os.path.join(args.output_dir, npz_fn), **{args.npz_key: new_arr})
        print("  done " + npz_fn + " missing=" + str(missing)); sys.stdout.flush()

    if not args.keep_work_dir:
        shutil.rmtree(args.work_dir)
    print("\n[done] " + args.output_dir)

if __name__ == "__main__":
    main()
