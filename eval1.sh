python train.py --dataset UVG --model_type HDNeRV2 --model_size S \
        --eval_only --weight checkpoints/HDNeRV2/S.pth -b 2 --method normal -p 500

python train.py --dataset UVG --model_type HDNeRV2 --model_size M \
        --eval_only --weight checkpoints/HDNeRV2/M.pth -b 2 --method normal -p 500

python train.py --dataset UVG --model_type HDNeRV2 --model_size L \
        --eval_only --weight checkpoints/HDNeRV2/L.pth -b 2 --method normal -p 500