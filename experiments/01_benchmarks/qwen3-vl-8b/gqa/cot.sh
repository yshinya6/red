PATTERN=gqa-qwen3vl-8b-cot
TASK=gqa
MODEL=Qwen/Qwen3-VL-8B-Instruct
IMGDIR=/dataset/gqa/images
QUESTION=data/gqa/test.json
RATINALE=cot

python evaluation/cot.py --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK --answers-file ./test/${PATTERN}.jsonl --rationale_type $RATINALE --verbose
