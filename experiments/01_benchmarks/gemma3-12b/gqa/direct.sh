PATTERN=gqa-gemma3-12b-direct
TASK=gqa
MODEL=google/gemma-3-12b-it
IMGDIR=/dataset/gqa/images
QUESTION=data/gqa/test.json

python evaluation/direct.py --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK --answers-file ./test/${PATTERN}.jsonl --verbose
