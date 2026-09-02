#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert raw object-detection annotations into the unified COCO-VID style
JSON that this repository consumes (see datasets/vid_multi.py / vid_single.py).

Supported source formats
------------------------
1) ImageNet VID  (annotation = one XML per frame, VOC-like layout)
     <ann_root>/<...>/<video>/000000.xml
   * the <name> field inside each <object> is an ImageNet wnid, mapped to
     category id 1..30 with the same table used by mmtracking
     (CLASSES / CLASSES_ENCODES below).
2) VisDrone      (annotation = one TXT per frame)
     each line: x1,y1,w,h,score,category,truncation,occlusion
   * category 0 (ignore region) is dropped; category 11 (others) is dropped
     by default (--keep-others to keep it).  Categories keep official ids 1..10.

Both are written to the SAME JSON schema (videos/images/annotations/
categories) so you can train/evaluate either dataset with this repo.

Usage examples
--------------
# ImageNet VID
python tools/convert_to_vid_json.py --dataset imagenet_vid \
    --ann-dir <ILSVRC2015>/Annotations/VID/val \
    --img-root <ILSVRC2015>/Data/VID/val \
    --out annotations/imagenet_vid_val.json

python tools/convert_to_vid_json.py --dataset imagenet_vid \
    --ann-dir <ILSVRC2015>/Annotations/VID/train \
    --img-root <ILSVRC2015>/Data/VID/train \
    --out annotations/imagenet_vid_train.json

# VisDrone-VID (annotations/<seq>/0000001.txt <-> sequences/<seq>/0000001.jpg)
python tools/convert_to_vid_json.py --dataset visdrone \
    --ann-dir <VisDrone2019-VID-train>/annotations \
    --img-root <VisDrone2019-VID-train>/sequences \
    --out annotations/visdrone_vid_train.json

# VisDrone-DET (flat annotations/*.txt, one image per video)
python tools/convert_to_vid_json.py --dataset visdrone \
    --ann-dir <VisDrone2019-DET-val>/annotations \
    --img-root <VisDrone2019-DET-val>/images \
    --out annotations/visdrone_det_val.json

Note on the file_name field: file_name is written relative to --img-root.
Point the img_folder of the dataset you build (datasets/vid_*.py PATHS) at
that --img-root directory (or symlink it there) so images resolve correctly.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

try:
    import xml.etree.cElementTree as ET
except ImportError:  # pragma: no cover
    import xml.etree.ElementTree as ET

logger = logging.getLogger("convert_to_vid_json")


# --------------------------------------------------------------------------- #
# ImageNet VID: 30 classes, order (and wnids) identical to mmtracking so the
# produced json is interchangeable with the official converted files.
# --------------------------------------------------------------------------- #
VID_CLASSES = (
    "airplane", "antelope", "bear", "bicycle", "bird",
    "bus", "car", "cattle", "dog", "domestic_cat",
    "elephant", "fox", "giant_panda", "hamster", "horse",
    "lion", "lizard", "monkey", "motorcycle", "rabbit",
    "red_panda", "sheep", "snake", "squirrel", "tiger",
    "train", "turtle", "watercraft", "whale", "zebra",
)
VID_CLASSES_ENCODES = (
    "n02691156", "n02419796", "n02131653", "n02834778", "n01503061",
    "n02924116", "n02958343", "n02402425", "n02084071", "n02121808",
    "n02503517", "n02118333", "n02510455", "n02342885", "n02374451",
    "n02129165", "n01674464", "n02484322", "n03790512", "n02324045",
    "n02509815", "n02411705", "n01726692", "n02355227", "n02129604",
    "n04468005", "n01662784", "n04530566", "n02062744", "n02391049",
)
VID_NAME_TO_ID = {wnid: i for i, wnid in enumerate(VID_CLASSES_ENCODES, 1)}

# --------------------------------------------------------------------------- #
# VisDrone: official category ids/names.  0 = ignore-region (always dropped),
# 11 = others (dropped unless --keep-others).
# --------------------------------------------------------------------------- #
VISDRONE_NAMES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
    11: "others",
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def natural_key(path: Path):
    m = re.search(r"\d+", path.stem)
    return (int(m.group()), path.name) if m else (10 ** 12, path.name)


def find_image(img_root: Path, ann_rel_dir: str, stem: str):
    """Locate the image that matches an annotation file and return its posix
    path relative to img_root, or None if not found."""
    for ext in ("jpg", "jpeg", "png"):
        for name in (f"{stem}.{ext}", f"{stem}.{ext.upper()}"):
            p = img_root / ann_rel_dir / name
            if p.is_file():
                return p.relative_to(img_root).as_posix()
    return None


def read_image_size(img_path: Path):
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.size
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# group discovery: directories that directly contain the annotation files
# --------------------------------------------------------------------------- #
def discover_annotation_groups(ann_dir: str, suffix: str):
    """Return a list of (video_name, rel_dir, [frame_path, ...]) sorted.

    - Nested layout (e.g. ImageNet VID train: .../<video>/*.xml or
      VisDrone VID: annotations/<video>/*.txt) -> each folder holding files
      is one video.
    - Flat layout (e.g. VisDrone DET: annotations/*.txt) -> every file is
      treated as a single-frame video so temporal neighbours are never
      sampled across unrelated images.
    """
    root = Path(ann_dir)
    assert root.is_dir(), f"annotation dir does not exist: {root}"
    groups = []
    for dirpath, _dirnames, filenames in os.walk(root):
        matched = sorted(
            (Path(dirpath) / f for f in filenames
             if f.lower().endswith(suffix)),
            key=natural_key)
        if not matched:
            continue
        rel = Path(dirpath).relative_to(root)
        if rel == Path("."):
            # flat: one file per (single-frame) video
            for fp in matched:
                groups.append((fp.stem, "", [fp]))
        else:
            groups.append((rel.as_posix(), rel.as_posix(), matched))
    groups.sort(key=lambda g: g[0])
    return groups


# --------------------------------------------------------------------------- #
# ImageNet VID XML parsing
# --------------------------------------------------------------------------- #
def parse_imagenet_video(xml_paths, ann_rel_dir, img_root, vid_id, counters):
    images, anns = [], []
    # trackid -> global instance id, valid within this video only
    track_map = {}
    for xml_path in xml_paths:
        root = ET.parse(xml_path).getroot()

        size = root.find("size")
        try:
            width = int(size.find("width").text)
            height = int(size.find("height").text)
        except Exception:
            width = height = 0

        stem = xml_path.stem
        file_name = None
        if img_root is not None:
            file_name = find_image(img_root, ann_rel_dir, stem)
        if file_name is None:
            file_name = f"{ann_rel_dir}/{stem}.JPEG" if ann_rel_dir else f"{stem}.JPEG"
            logger.warning("image not found for %s, using %s", xml_path,
                           file_name)

        counters["img_id"] += 1
        img_id = counters["img_id"]
        images.append(dict(
            id=img_id, file_name=file_name, width=width, height=height,
            video_id=vid_id, frame_id=len(images),
        ))

        for obj in root.findall("object"):
            name_el = obj.find("name")
            name = name_el.text if name_el is not None else ""
            if name not in VID_NAME_TO_ID:
                continue
            bndbox = obj.find("bndbox")
            try:
                x1 = float(bndbox.find("xmin").text)
                y1 = float(bndbox.find("ymin").text)
                x2 = float(bndbox.find("xmax").text)
                y2 = float(bndbox.find("ymax").text)
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            counters["ann_id"] += 1
            ann_id = counters["ann_id"]
            track_el = obj.find("trackid")
            if track_el is not None and track_el.text is not None:
                trackid = track_el.text
                if trackid not in track_map:
                    counters["instance_id"] += 1
                    track_map[trackid] = counters["instance_id"]
                instance_id = track_map[trackid]
            else:
                instance_id = ann_id

            occluded = obj.find("occluded")
            generated = obj.find("generated")
            w = x2 - x1
            h = y2 - y1
            anns.append(dict(
                id=ann_id, video_id=vid_id, image_id=img_id,
                category_id=VID_NAME_TO_ID[name], instance_id=instance_id,
                bbox=[x1, y1, w, h], area=w * h, iscrowd=False,
                occluded=bool(occluded is not None and occluded.text == "1"),
                generated=bool(generated is not None and generated.text == "1"),
            ))
    return images, anns


# --------------------------------------------------------------------------- #
# VisDrone TXT parsing
# --------------------------------------------------------------------------- #
def parse_visdrone_video(txt_paths, ann_rel_dir, img_root, vid_id, counters,
                         keep_others, drop_ignored):
    images, anns = [], []
    for txt_path in txt_paths:
        stem = txt_path.stem
        file_name = None
        if img_root is not None:
            file_name = find_image(img_root, ann_rel_dir, stem)
        if file_name is None:
            file_name = f"{ann_rel_dir}/{stem}.jpg" if ann_rel_dir else f"{stem}.jpg"
            logger.warning("image not found for %s, using %s", txt_path,
                           file_name)

        width = height = 0
        if img_root is not None:
            cand = img_root / file_name
            if cand.is_file():
                wh = read_image_size(cand)
                if wh:
                    width, height = wh

        counters["img_id"] += 1
        img_id = counters["img_id"]
        images.append(dict(
            id=img_id, file_name=file_name, width=width, height=height,
            video_id=vid_id, frame_id=len(images),
        ))

        with open(txt_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                tokens = [t for t in line.split(",") if t.strip()]
                try:
                    nums = [float(t) for t in tokens]
                except ValueError:
                    continue
                if len(nums) == 10:
                    # MOT-style: frame_index, target_id, x1, y1, w, h, score, cat, trunc, occ
                    x1, y1, w, h, score, cat, _trunc, _occ = nums[2:10]
                elif len(nums) >= 8:
                    # detection-style: x1, y1, w, h, score, cat, trunc, occ
                    x1, y1, w, h, score, cat, _trunc, _occ = nums[:8]
                else:
                    continue
                cat = int(cat)
                if cat == 0:                       # ignore region
                    continue
                if cat == 11 and not keep_others:
                    continue
                if drop_ignored and score <= 0:
                    continue
                if w <= 0 or h <= 0:
                    continue

                counters["ann_id"] += 1
                ann_id = counters["ann_id"]
                anns.append(dict(
                    id=ann_id, video_id=vid_id, image_id=img_id,
                    category_id=cat, instance_id=ann_id,   # txt has no track id
                    bbox=[x1, y1, w, h], area=w * h, iscrowd=False,
                ))
    return images, anns


# --------------------------------------------------------------------------- #
# top-level converters
# --------------------------------------------------------------------------- #
def convert_imagenet_vid(args):
    if args.img_root is None:
        logger.error("--img-root is required for imagenet_vid")
        sys.exit(1)
    groups = discover_annotation_groups(args.ann_dir, ".xml")
    if not groups:
        logger.error("no .xml annotation found under %s", args.ann_dir)
        sys.exit(1)
    logger.info("%d videos found", len(groups))

    videos, images, anns = [], [], []
    counters = dict(img_id=0, ann_id=0, instance_id=0)
    img_root = Path(args.img_root)
    for video_name, rel_dir, xml_paths in groups:
        kept = xml_paths[:: args.frame_step]
        vid_id = len(videos) + 1
        videos.append(dict(id=vid_id, name=video_name))
        im, an = parse_imagenet_video(kept, rel_dir, img_root, vid_id, counters)
        images.extend(im)
        anns.extend(an)
        if len(videos) % 500 == 0:
            logger.info("... %d videos / %d images", len(videos), len(images))

    categories = [
        dict(id=i, name=name, encode_name=wnid)
        for i, (name, wnid) in enumerate(
            zip(VID_CLASSES, VID_CLASSES_ENCODES), 1)
    ]
    return dict(videos=videos, images=images, annotations=anns,
                categories=categories)


def convert_visdrone(args):
    groups = discover_annotation_groups(args.ann_dir, ".txt")
    if not groups:
        logger.error("no .txt annotation found under %s", args.ann_dir)
        sys.exit(1)
    logger.info("%d videos found", len(groups))

    videos, images, anns = [], [], []
    counters = dict(img_id=0, ann_id=0)
    img_root = Path(args.img_root) if args.img_root else None
    for video_name, rel_dir, txt_paths in groups:
        kept = txt_paths[:: args.frame_step]
        vid_id = len(videos) + 1
        videos.append(dict(id=vid_id, name=video_name))
        im, an = parse_visdrone_video(
            kept, rel_dir, img_root, vid_id, counters,
            args.keep_others, args.drop_ignored)
        images.extend(im)
        anns.extend(an)
        if len(videos) % 500 == 0:
            logger.info("... %d videos / %d images", len(videos), len(images))

    cat_ids = [c for c in VISDRONE_NAMES if c != 0 and (c != 11 or args.keep_others)]
    categories = [dict(id=c, name=VISDRONE_NAMES[c]) for c in sorted(cat_ids)]
    return dict(videos=videos, images=images, annotations=anns,
                categories=categories)


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Convert ImageNet VID (XML) / VisDrone (TXT) annotations "
                    "into the unified COCO-VID JSON used by this repo.")
    parser.add_argument("--dataset", choices=["imagenet_vid", "visdrone"],
                        required=True, help="source annotation format")
    parser.add_argument("--ann-dir", required=True,
                        help="root dir that contains the annotation files "
                             "(XML for imagenet_vid, TXT for visdrone)")
    parser.add_argument("--img-root", default=None,
                        help="root dir of the images; file_name in the output "
                             "json is written relative to this dir")
    parser.add_argument("--out", required=True,
                        help="output json path (e.g. annotations/xxx.json)")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="keep every Nth frame of each video (default 1 = "
                             "all frames). Larger values subsample, e.g. 15.")
    parser.add_argument("--keep-others", action="store_true",
                        help="[visdrone] keep category 11 'others' "
                             "(dropped by default)")
    parser.add_argument("--no-drop-ignored", dest="drop_ignored",
                        action="store_false",
                        help="[visdrone] do not drop lines with score<=0")
    parser.add_argument("--indent", type=int, default=None,
                        help="json indent (omit for compact output)")
    parser.set_defaults(drop_ignored=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout)

    if args.frame_step < 1:
        parser.error("--frame-step must be >= 1")

    convert = (convert_imagenet_vid if args.dataset == "imagenet_vid"
               else convert_visdrone)
    out_data = convert(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out_data, fh, indent=args.indent)

    logger.info("done: %s", out_path)
    logger.info("  videos=%d images=%d annotations=%d categories=%d",
                len(out_data["videos"]), len(out_data["images"]),
                len(out_data["annotations"]), len(out_data["categories"]))
    # sanity: every image / annotation knows its video
    assert all("video_id" in im for im in out_data["images"])
    assert all("video_id" in an for an in out_data["annotations"])


if __name__ == "__main__":
    main()
