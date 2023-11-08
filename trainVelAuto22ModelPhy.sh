mkdir ./autoencoderccL1
# rm -rf /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/checkpoints/MarmousiOpenFWI22/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/utig5/disk/fs4/abdo/PhysicsMLFWI/PhysicsBasedFWI/MarmousiN/ --name MarmousiOpenFWIAutoccL1 --model Auto22 --direction AtoB --input_nc 18 --output_nc 1 --display_port 9998 --n_epochs 5000 --n_epochs_decay 0 --batch_size 8 --gpu_ids 0 --no_html --display_freq 1 --print_freq 1 --lr 0.001 --verbose --save_epoch_freq 50
