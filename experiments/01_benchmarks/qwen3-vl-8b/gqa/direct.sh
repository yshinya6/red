PATTERN=gqa-qwen3vl-8b-direct
TASK=gqa
MODEL=Qwen/Qwen3-VL-8B-Instruct
IMGDIR=/dataset/gqa/images
QUESTION=data/gqa/test.json

python evaluation/direct.py --model-path $MODEL --image-folder $IMGDIR --question-file $QUESTION --evaluation $TASK --answers-file ./test/${PATTERN}.jsonl --verbose
