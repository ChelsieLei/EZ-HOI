DIR=../../data/mscoco2014
TRAIN=train2014
VAL=val2014
ANNO=annotations

if [ -d $DIR/$TRAIN ];
then
    echo "$DIR/$TRAIN already exists."
else
    mkdir $DIR
    wget http://images.cocodataset.org/zips/train2014.zip
    echo "Unzipping training set...this will take a few minutes"
    unzip train2014.zip &>train
    echo "Cleaning up..."
    mv $TRAIN $DIR/$TRAIN
    rm train
    rm train2014.zip
    echo "Done"
fi

if [ -d $DIR/$VAL ];
then
    echo "$DIR/$VAL already exists."
else
    wget http://images.cocodataset.org/zips/val2014.zip
    echo "Unzipping validation set...this should be done in a minute"
    unzip val2014.zip &>val
    echo "Cleaning up..."
    mv $VAL $DIR/$VAL
    rm val
    rm val2014.zip
    echo "Done"
fi

if [ -d $DIR/$ANNO ];
then
    echo "$DIR/$ANNO already exists."
else
    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    echo "Unzipping annotation files...this won't take long"
    unzip annotations_trainval2014.zip &>anno
    echo "Cleaning up"
    mv $ANNO $DIR/$ANNO
    rm anno
    rm annotations_trainval2014.zip
    echo "Done"
fi
