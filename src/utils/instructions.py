################################################################################################
# Instructions for Direct Answer
################################################################################################

DIRECT_ANSWER_INST = """
Instruction: Please directly output only answer without any reasoning.

"""


################################################################################################
# Instructions for CoT
################################################################################################

COT_RATIONALE_INST = """
## Instruction

For the provided image and its associated question, please provide detailed reasoning required to output the answer. To output the description, please identify each element related to the question in the image and explain how each contributes to answer the question. Please compose your output with short and clear sentences. 

## Response

[Provide only the necessary and sufficient description here. Be specific and concise.]
"""

COT_ANSWER_INST = """
Instruction: Based on the associated reasoning, answer the following question.
"""

COT_HEADER = "Reasoning: "

################################################################################################
# Instructions for CCoT
################################################################################################

CCOT_RATIONALE_INST = """
## Instruction

Given an image and its associated question, describe a **necessary and sufficient** visual scene graph in JSON format.  The scene graph can include the following:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Object relationships that are relevant to answering the question.

## Response

[Provide only the necessary and sufficient visual scene graph here. Be specific and concise.]
"""

CCOT_ANSWER_INST = """
Instruction: Based on the scene graph, answer the following question. 
"""

CCOT_HEADER = "Scene Graph: "

INSTRUCTION_TRIPLE = {
    # [RATIONALE_INST, ANSWER_INST, HEADER]
    "direct": [None, DIRECT_ANSWER_INST, None],
    "cot": [COT_RATIONALE_INST, COT_ANSWER_INST, COT_HEADER],
    "ccot": [CCOT_RATIONALE_INST, CCOT_ANSWER_INST, CCOT_HEADER],
}
