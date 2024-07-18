PYTHON="/home/wabdul/miniconda3/bin/python"
SCRIPTS="/data/wabdul/compressTraj/scripts"

DEFFNM="t4l_l99a_bnz"

PREFIX="/data/wabdul/compressTraj/github/example"
REFFILE1=$PREFIX/$DEFFNM.pdb
REFFILE2=$PREFIX/"$DEFFNM"_heavy.pdb
TRAJFILE1=$PREFIX/"$DEFFNM".xtc
TRAJFILE2=$PREFIX/"$DEFFNM"_decompressed.xtc
MODEL=$PREFIX/"$DEFFNM"_model.pt
SCALER=$PREFIX/"$DEFFNM"_scaler.pkl
COMPRESSED=$PREFIX/"$DEFFNM"_compressed.pkl

$PYTHON $SCRIPTS/prepare_data.py --reffile $REFFILE1 --trajfile $TRAJFILE1 --prefix $PREFIX/"$DEFFNM"
$PYTHON $SCRIPTS/train_ae.py --train_loader $PREFIX/"$DEFFNM"_train.pkl -v $PREFIX/"$DEFFNM"_val.pkl -p $PREFIX/"$DEFFNM" -e 200
$PYTHON $SCRIPTS/compress.py -m $MODEL -r $REFFILE1 -t $TRAJFILE1 -o $COMPRESSED
$PYTHON $SCRIPTS/decompress.py -m $MODEL -s $SCALER -r $REFFILE1 -o $TRAJFILE2 -c $COMPRESSED -p $PREFIX/"$DEFFNM"_padding.txt
$PYTHON $SCRIPTS/rmsd_frames.py -r1 $REFFILE1 -t1 $TRAJFILE1 -r2 $REFFILE2 -t2 $TRAJFILE2 -o $PREFIX/"$DEFFNM"_rmsd.txt
