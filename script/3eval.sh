python train.py --model_type HDNeRV3 --model_size S \
        --eval_only --weight checkpoints/HDNeRV3/S.pth -b 8 -p 500 --dump_images

python train.py --model_type HDNeRV3 --model_size M \
        --eval_only --weight checkpoints/HDNeRV3/M.pth -b 8 -p 500 --dump_images

python train.py --model_type HDNeRV3 --model_size L \
        --eval_only --weight checkpoints/HDNeRV3/L.pth -b 8 -p 500 --dump_images

python train.py --model_type HDNeRV3 --model_size S \
        --eval_only --weight checkpoints/HDNeRV3/S.pth -b 8 -p 500 --dump_images --method normal

python train.py --model_type HDNeRV3 --model_size M \
        --eval_only --weight checkpoints/HDNeRV3/M.pth -b 8 -p 500 --dump_images --method normal

python train.py --model_type HDNeRV3 --model_size L \
        --eval_only --weight checkpoints/HDNeRV3/L.pth -b 8 -p 500 --dump_images --method normal

python train.py --model_type HDNeRV3 --model_size S \
        --eval_only --weight checkpoints/HDNeRV3/S.pth -b 8 -p 500 --dump_images --method cabac

python train.py --model_type HDNeRV3 --model_size M \
        --eval_only --weight checkpoints/HDNeRV3/M.pth -b 8 -p 500 --dump_images --method cabac

python train.py --model_type HDNeRV3 --model_size L \
        --eval_only --weight checkpoints/HDNeRV3/L.pth -b 8 -p 500 --dump_images --method cabac

python train.py --model_type HDNeRV3 --model_size S \
        --eval_only --weight checkpoints/HDNeRV3/S.pth -b 8 -p 500 --dump_images --method cabac --qp -24

python train.py --model_type HDNeRV3 --model_size M \
        --eval_only --weight checkpoints/HDNeRV3/M.pth -b 8 -p 500 --dump_images --method cabac --qp -24

python train.py --model_type HDNeRV3 --model_size L \
        --eval_only --weight checkpoints/HDNeRV3/L.pth -b 8 -p 500 --dump_images --method cabac --qp -24
