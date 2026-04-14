PATTERN=gqa-qwen3vl-8b-red
TASK=gqa
MODEL=Qwen/Qwen3-VL-8B-Instruct
IMGDIR=/dataset/gqa/images
QUESTION=data/gqa/test.json
RATINALE=cot
LAMBDA=0.1

python evaluation/red.py --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK --answers-file ./test/${PATTERN}.jsonl --rationale_type $RATINALE --lambda_rat $LAMBDA --verbose
