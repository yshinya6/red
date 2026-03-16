PATTERN=gqa-qwen2.5vl-8b-direct
TASK=gqa
MODEL=Qwen/Qwen2.5-VL-8B-Instruct
IMGDIR=data/gqa/images
QUESTION=data/gqa/test.json

python evaluation/direct.py --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK --answers-file ./test/${PATTERN}.jsonl --verbose
