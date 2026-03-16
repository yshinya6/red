PATTERN=gqa-qwen2.5vl-8b-red
TASK=gqa
MODEL=Qwen/Qwen2.5-VL-8B-Instruct
IMGDIR=data/gqa/images
QUESTION=data/gqa/test.json
RATINALE=cot
LAMBDA=0.5

python evaluation/cot.py --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK --answers-file ./test/${PATTERN}.jsonl --rationale_type $RATINALE --lambda_rat $LAMBDA --verbose
