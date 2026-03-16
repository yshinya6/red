PATTERN=gqa-gemma3-12b-cot
TASK=gqa
MODEL=google/gemma-3-12b-it
IMGDIR=data/gqa/images
QUESTION=data/gqa/test.json
RATINALE=cot

python evaluation/cot.py --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK --answers-file ./test/${PATTERN}.jsonl --rationale_type $RATINALE --verbose
