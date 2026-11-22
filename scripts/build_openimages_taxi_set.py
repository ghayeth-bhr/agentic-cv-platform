"""Build the Open Images V7 taxi benchmark set as a COCO dataset.

Keeps validation and test images (whose masks were drawn entirely by hand)
where every taxi box has a mask and none is a "group of" box, so a taxi the
model finds is never missing from the ground truth. About 210 MB of mask
archives are downloaded; the result is ~31 MB.

    python scripts/build_openimages_taxi_set.py <cache-dir> <output-dir>

The BotFlow benchmark then removes one mislabeled mask (a police car labeled as
a taxi in 134f39599bbc5848.jpg) after review in the gallery; see
benchmarks/README.md.
"""
import csv, collections, io, json, os, sys, urllib.request, zipfile
import numpy as np, cv2

META, OUT = sys.argv[1], sys.argv[2]          # folder with the CSVs, output folder
TAXI = '/m/0pg52'
os.makedirs(os.path.join(OUT, 'images'), exist_ok=True)
os.makedirs(META, exist_ok=True)

def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()

for split in ('validation', 'test'):
    for kind, name in (('object-segmentation', 'seg'), ('bbox', 'bbox')):
        target = os.path.join(META, f'{split}-{name}.csv')
        if not os.path.exists(target):
            print('downloading', split, kind, flush=True)
            data = fetch(f'https://storage.googleapis.com/openimages/v5/{split}-annotations-{kind}.csv')
            open(target, 'wb').write(data)

rows = collections.defaultdict(list); keep = {}
for split in ('validation', 'test'):
    boxes = collections.Counter(); group = set(); masks = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(META, f'{split}-seg.csv'))):
        if r['LabelName'] == TAXI: masks[r['ImageID']].append(r)
    for r in csv.DictReader(open(os.path.join(META, f'{split}-bbox.csv'))):
        if r['ImageID'] in masks and r['LabelName'] == TAXI:
            boxes[r['ImageID']] += 1
            if r['IsGroupOf'] == '1': group.add(r['ImageID'])
    for i, ms in masks.items():
        if len(ms) >= boxes[i] and i not in group:
            keep[i] = split; rows[i] = ms
print('images', len(keep), 'masks', sum(len(v) for v in rows.values()))

# masks: one zip per split and first hex char of the image id
wanted = {r['MaskPath'] for i in keep for r in rows[i]}
pngs = {}
for split in ('validation', 'test'):
    for c in sorted({i[0] for i, s in keep.items() if s == split}):
        z = zipfile.ZipFile(io.BytesIO(fetch(f'https://storage.googleapis.com/openimages/v5/{split}-masks/{split}-masks-{c}.zip')))
        for n in z.namelist():
            if os.path.basename(n) in wanted:
                pngs[os.path.basename(n)] = z.read(n)
        print('masks zip', split, c, 'found so far', len(pngs), flush=True)

images, anns = [], []
for k, (iid, split) in enumerate(sorted(keep.items()), 1):
    data = fetch(f'https://s3.amazonaws.com/open-images-dataset/{split}/{iid}.jpg')
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None: print('skip unreadable', iid); continue
    h, w = img.shape[:2]
    open(os.path.join(OUT, 'images', f'{iid}.jpg'), 'wb').write(data)
    images.append({'id': k, 'file_name': f'{iid}.jpg', 'width': w, 'height': h, 'oi_split': split})
    for r in rows[iid]:
        m = cv2.imdecode(np.frombuffer(pngs[r['MaskPath']], np.uint8), cv2.IMREAD_GRAYSCALE)
        m = (cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = [c.reshape(-1).astype(float).tolist() for c in cs if len(c) >= 3]
        if not polys: continue
        x, y, bw, bh = cv2.boundingRect(m)
        anns.append({'id': len(anns) + 1, 'image_id': k, 'category_id': 1, 'segmentation': polys,
                     'area': float(m.sum()), 'bbox': [x, y, bw, bh], 'iscrowd': 0})
    if k % 10 == 0: print('images', k, flush=True)

coco = {'info': {'description': 'Open Images V7 taxi benchmark set: validation+test images where every taxi has a hand-drawn mask',
                 'url': 'https://storage.googleapis.com/openimages/web/index.html',
                 'license_note': 'Annotations CC BY 4.0 (Google); images listed as CC BY 2.0 by their Flickr authors'},
        'images': images, 'annotations': anns, 'categories': [{'id': 1, 'name': 'taxi'}]}
json.dump(coco, open(os.path.join(OUT, 'annotations.coco.json'), 'w'))
print('done', len(images), 'images', len(anns), 'annotations')
