# Detection Utilities

## Generate detection using Faster R-CNN
```bash
python preprocessing.py --partition trainval
```

A Faster R-CNN model pre-trained on MS COCO will be used by default to generate detections. Use the argument `--partition` to specify the subset to run the detector on. To run a Faster R-CNN model with different weights, use the argument `--ckpt-path` to load the model from specified checkpoint. Run `python preprocessing.py --help` to find out more about post-processing options. The generated detections will be saved in a directory named after the parition e.g. `trainval`.

## Visualise detections

```bash
python visualise.py --image-idx 100
```

Visualise detections for an image. Use argument `--partition` to specify the subset. __Note__ that it is assumed the directory where detections are saved has the same name as the partition. To select a specific image, use the argument `--image-idx`.

## Evaluate detections

```bash
python eval_detections.py --detection-root ./test
```

Evaluate the mAP of the detections against the ground truth object detections of HICO-DET. Use the argument `--partition` to specify the subset to evaluate against. The default is `test`. Use the argument `--detection-root` to point to the directory where detection results are saved. Run `python eval_detections.py --help` to find out more about post-processing options.