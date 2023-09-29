CUDA_VISIBLE_DEVICES=1  python train.py --model_type HDNeRV2 --model_size L \
        --eval_only --weight checkpoints/HDNeRV2/L.pth -b 4 -p 500 --method cabac --qp -22

CUDA_VISIBLE_DEVICES=1  python train.py --model_type HDNeRV2 --model_size L \
        --eval_only --weight checkpoints/HDNeRV2/L.pth -b 4 -p 500 --method cabac --qp -26

CUDA_VISIBLE_DEVICES=1  python train.py --model_type HDNeRV2 --model_size L \
        --eval_only --weight checkpoints/HDNeRV2/L.pth -b 4 -p 500 --method cabac --qp -30

CUDA_VISIBLE_DEVICES=1 python train.py --model_type HDNeRV2 --model_size M \
        --eval_only --weight checkpoints/HDNeRV2/M.pth -b 4 -p 500 --method cabac --qp -22

CUDA_VISIBLE_DEVICES=1 python train.py --model_type HDNeRV2 --model_size M \
        --eval_only --weight checkpoints/HDNeRV2/M.pth -b 4 -p 500 --method cabac --qp -26
        
CUDA_VISIBLE_DEVICES=1 python train.py --model_type HDNeRV2 --model_size M \
        --eval_only --weight checkpoints/HDNeRV2/M.pth -b 4 -p 500 --method cabac --qp -30

CUDA_VISIBLE_DEVICES=1 python train.py --model_type HDNeRV2 --model_size S \
        --eval_only --weight checkpoints/HDNeRV2/S.pth -b 4 -p 500 --method cabac --qp -22

CUDA_VISIBLE_DEVICES=1 python train.py --model_type HDNeRV2 --model_size S \
        --eval_only --weight checkpoints/HDNeRV2/S.pth -b 4 -p 500 --method cabac --qp -26

CUDA_VISIBLE_DEVICES=1 python train.py --model_type HDNeRV2 --model_size S \
        --eval_only --weight checkpoints/HDNeRV2/S.pth -b 4 -p 500 --method cabac --qp -30

