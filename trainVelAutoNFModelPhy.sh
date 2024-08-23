
mkdir ./MarmousiNF13
rm -rf /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/checkpoints/MarmousiNF13/*.txt
no_proxy=localhost python trainValLatent4dVel3.py --dataroot /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/MarmousiBig6 --name MarmousiNF13 --model AutoNF --direction AtoB --input_nc 24 --output_nc 1 --display_port 8881 --n_epochs 1000 --n_epochs_decay 3000  --batch_size 1 --gpu_ids 1 --no_html --display_freq 1 --print_freq 1 --lr 0.0001 --verbose --save_epoch_freq 50 
