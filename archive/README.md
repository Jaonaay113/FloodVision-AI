# Combined flood segmentation dataset

One folder that holds the Kaggle Flood Area Segmentation set, FloodNet (extra training data)
and AIFloodSense (external test set), in the same layout the notebooks already use:

```
dataset_combined/
  Image/<source>_<id>.jpg
  Mask/<source>_<id>.png      binary: 0 = background, 255 = flood water
  manifest.csv                one row per image/mask pair
```

The existing `FloodDataset` loads these files unchanged (its `mask > 127` rule works on 0/255 masks).

## Contents (built 2026-09-21)

| Source | Role | Pairs | Notes |
|---|---|---|---|
| Kaggle Flood Area Segmentation | `kaggle` | 290 | Images byte-identical to `dataset/`; masks binarised at >127 and resized to their image where sizes differed, exactly as `FloodDataset` does at load time |
| FloodNet v1.0 train+val | `extra_train` | **1,825** of 1,895 train+val | The complete official release; 856 have no flood water, which is what makes false alarms measurable |
| FloodNet v1.0 test | `floodnet_test` | 424 of 448 | Held out; not used by the default splits |
| AIFloodSense test | `external_test` | 94 | Zero-shot external test set; never train on it |
| AIFloodSense train | `external_pool` | 376 | Also never trained on; together with the test split gives 467 unique external images |

Total **3,009 pairs, 1.0 GB**. Cross-source duplicate check (pHash distance ≤ 8 across all
images): **0 matches**. A further 94 FloodNet images were converted but flagged
`ambiguous_label` and left out (see below).

## Water definition per source

- **Kaggle:** the dataset's white (water) class.
- **FloodNet:** classes `water` and `road-flooded` from the official full-resolution index masks
  (`*-label-img`). Not water: `pool` and `building-flooded` (the building footprint). The
  1024×1024 colour masks are a coarser rendering and are used only to confirm the class order;
  where the two differ, the index labels are more complete (compared on images 6694 and 10833).
  103 colour masks use colours outside the palette — (0,163,255) covers water on 88 of them and
  (120,120,70) a fenced structure on 15 — while the index labels mark both as background. Those
  images therefore contain unlabelled water, so the 94 whose undocumented share exceeds 1%
  are flagged `ambiguous_label` in `dataset_extra/FloodNet/manifest.csv` and excluded from
  the merged set.
- **AIFloodSense:** mask value 255 (`Flood`) only. Sky (170) and buildings (85) are not water.

All three treat buildings standing in water and the sky as non-water, so the definitions agree.

## manifest.csv columns

| Column | Meaning |
|---|---|
| `image`, `mask` | file names in `Image/` and `Mask/` |
| `source` | `kaggle`, `floodnet` or `afs` |
| `source_image` | original file name in the source dataset |
| `file_format` | real encoding; 45 Kaggle files named `.jpg` are WebP/PNG/GIF (read fine by OpenCV/PIL) |
| `role` | `kaggle`, `extra_train`, `floodnet_test`, `external_test`, `external_pool` |
| `source_split` | the source's own split (Kaggle: the notebooks' 70/15/15 split) |
| `kaggle_fold`, `kaggle_group` | duplicate-aware 5-fold assignment and duplicate group (Kaggle only, else -1) |
| `duplicate_of` | set on the redundant copy of a within-source duplicate; such copies are left out of val/test/external |
| `width`, `height`, `flood_fraction` | stored image size and share of water pixels |
| `exclude`, `exclude_reason` | cross-source near-duplicates, always dropped (none at present) |

## Using it in the notebooks

Replace the split in the "Data Cleaning and Path Setup" cell with:

```python
from flood_data import combined_split, source_balanced_sampler

s = combined_split(fold=0, use_floodnet=True, external='test')   # fold=None -> original 70/15/15 split
train_imgs, train_masks = s['train']
val_imgs, val_masks = s['val']
test_imgs, test_masks = s['test']
ext_imgs, ext_masks = s['external']        # AIFloodSense, evaluate only
```

and in the DataLoader cell use the balanced sampler for training (FloodNet otherwise outnumbers
Kaggle about 10:1 and has far less water per image, 14% vs 41%):

```python
train_loader = DataLoader(FloodDataset(train_imgs, train_masks, train_transform), batch_size=8,
                          sampler=source_balanced_sampler(train_imgs))
ext_loader = DataLoader(FloodDataset(ext_imgs, ext_masks, val_transform), batch_size=8, shuffle=False)
```

Set sizes: original split 2,028 / 42 / 39 (train / val / test) and 1,999 / 55 / 54 per fold, both
with 94 external images. `use_floodnet=False` gives 203 and 174 training images. The original split
drops the 5 test and 1 validation images that duplicate other images, so old and new results can
be compared on leak-free data.

## FloodNet is complete

All 2,343 FloodNet photos come from the official release, `FloodNet-Supervised_v1.0.zip`
(13.0 GB, every member CRC-checked). The photos are read straight out of the zip, so nothing
is extracted except the label masks, which live in `dataset_extra/_floodnet_raw` (87 MB).
2,249 of the 2,343 converted; the rest are the `ambiguous_label` images described above.

```
python prepare_floodnet.py --zip FloodNet-Supervised_v1.0.zip \
    --labels dataset_extra/_floodnet_raw --colormasks ColorMasks-FloodNetv1.0.zip
python build_combined_dataset.py
```

The earlier build held only the 533 photos that a Kaggle mirror served; `download_floodnet.py`
is kept for that case but is no longer needed.

## Known label issues (left as published)

- Kaggle: 14 duplicate groups with inconsistent masks (median IoU 0.83 between copies); see the paper's audit.
- FloodNet: some masks are coarse, e.g. 7283 labels 99% of the frame as water, houses included. 24 of 2,343 masks are >95% water.
- FloodNet: 103 masks leave water unlabelled (see above); the 94 above the 1% threshold are excluded.
- AIFloodSense: 3 near-duplicate pairs (076/090, 189/252, 395/468), one crossing its own train/test split.

## Licences and citations

- Kaggle Flood Area Segmentation: see the Kaggle dataset page (licence to be confirmed).
- FloodNet: Community Data License Agreement (permissive). Cite Rahnemoonfar et al., IEEE Access 9 (2021) 89644–89654.
- AIFloodSense: CC BY 4.0 (dataset README). Cite Simantiris et al., arXiv:2512.17432 (2025).

## Rebuilding from scratch

```
python prepare_aifloodsense.py archive.zip                     # -> dataset_external/AIFloodSense
python prepare_floodnet.py --images ... --labels ... --colormasks ...   # -> dataset_extra/FloodNet
python build_combined_dataset.py                               # -> dataset_combined (keep this README)
python summarize_dataset.py                                    # stats + sample montage
python evaluate_external.py && python paired_stats.py          # zero-shot results of saved checkpoints
```
