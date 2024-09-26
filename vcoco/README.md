# V-COCO
Utilities for the human-object interaction detection dataset [V-COCO](https://arxiv.org/abs/1505.04474)

## Supported Utilities

- [x] [A command-line style dataset navigator](https://github.com/fredzzhang/vcoco/tree/main/utilities#dataset-navigator)
- [x] [Large-scale visualisation in web page](https://github.com/fredzzhang/vcoco/tree/main/utilities#generate-and-visaulise-box-pairs-in-large-scales)
- [x] [Generate object detections with Faster R-CNN](https://github.com/fredzzhang/vcoco/tree/main/detections#generate-detection-using-faster-r-cnn)
- [x] [Visualise detected objects](https://github.com/fredzzhang/vcoco/tree/main/detections#visualise-detections)
- [x] [Evaluate object detections](https://github.com/fredzzhang/vcoco/tree/main/detections#evaluate-detections)

## Installation Instructions
1. Download the repo with `git clone https://github.com/fredzzhang/vcoco.git`
2. Download the `train2014` and `val2014` partitions of the [COCO dataset](https://cocodataset.org/#download)
    1. If you have not downloaded the dataset before, run the following script
    ```bash
    cd /path/to/vcoco
    bash download.sh
    ```
    2. If you have previsouly downloaded the dataset, simply create a soft link. Note that 
    ```bash
    cd /path/to/vcoco
    ln -s /path/to/coco ./mscoco2014
    ```
3. Install the lightweight deep learning library [Pocket](https://github.com/fredzzhang/pocket)
4. Make sure the environment you created for Pocket is activated. You are good to go!

## Miscellaneous
* The implementation of the dataset class can be found in [vcoco.py](https://github.com/fredzzhang/vcoco/blob/main/vcoco.py)
* The script that generated the annotation files can be found in [utilities/generate_annotations.py](https://github.com/fredzzhang/vcoco/blob/main/utilities/generate_annotations.py)

## License

[MIT License](./LICENSE)
