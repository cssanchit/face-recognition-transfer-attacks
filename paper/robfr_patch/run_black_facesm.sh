# FGSM + FaceSM
python -m RobFR.benchmark.FGSM_black --dataset lfw --model ArcFace --goal dodging --distance l2 --eps 4 --batch_size 40 --facesm --source-lambda 0.20 --output output/lfw-FGSM-FaceSM-l2-dodging-ArcFace

# BIM + FaceSM
python -m RobFR.benchmark.BIM_black --dataset lfw --model ArcFace --goal dodging --distance l2 --eps 4 --iters 100 --batch_size 40 --facesm --source-lambda 0.20 --output output/lfw-BIM-FaceSM-l2-dodging-ArcFace

# MIM + FaceSM
python -m RobFR.benchmark.MIM_black --dataset lfw --model ArcFace --goal dodging --distance l2 --eps 4 --iters 20 --mu 1.0 --batch_size 30 --facesm --source-lambda 0.20 --output output/lfw-MIM-FaceSM-l2-dodging-ArcFace

# CIM + FaceSM
python -m RobFR.benchmark.CIM_black --dataset lfw --model ArcFace --goal dodging --distance l2 --eps 4 --iters 20 --mu 1.0 --length 10 --batch_size 30 --facesm --source-lambda 0.20 --output output/lfw-CIM-FaceSM-l2-dodging-ArcFace
