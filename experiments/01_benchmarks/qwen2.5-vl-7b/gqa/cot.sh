PATTERN=gqa-qwen2.5vl-7b-cot
TASK=gqa
MODEL=Qwen/Qwen2.5-VL-7B-Instruct
IMGDIR=data/gqa/images
QUESTION=data/gqa/test.json
RATINALE=cot

python evaluation/cot.py --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK --answers-file ./test/${PATTERN}.jsonl --rationale_type $RATINALE --verbose
