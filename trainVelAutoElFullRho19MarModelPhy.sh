rm -rf ./AutoElMarmousi3
mkdir ./AutoElMarmousi3
rm -rf /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/checkpoints/AutoElMarmousi3/*.txt
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/marmousielastic --name AutoElMarmousi3 --model AutoElFullRhoMar22 --direction AtoB --input_nc 46 --output_nc 1 --display_port 9998 --n_epochs 4000  --n_epochs_decay 2500 --batch_size 1 --gpu_ids 5 --no_html --display_freq 1 --print_freq 1 --lr 0.005 --verbose --save_epoch_freq 20 --init_type kaiming
