import os
import glob
import json

options = """
A: The response attempts exclusively to address Question 1.
B: The response attempts exclusively to address Question 2.
C: The response attempts to address both Question 1 and Question 2.
D: The response refuses to provide an answer.
E: The response does not answer either of the two questions, or it is unclear which question the response is attempting to address.
"""

current_directory = os.getcwd()
pattern = os.path.join(current_directory, 'group*.json')

for file_path in glob.glob(pattern):
    print(file_path)  # 或者执行其他操作
    with open(file_path, "r") as file:
        data = json.load(file)
    for idx, dp in enumerate(data):
        context = dp["context"]
        question1 = dp["question 1"]
        answer1 = dp["answer 1"][0]
        question2 = dp["question 2"]
        answer2 = dp["answer 2"][0]
        responses = dp["responses"]
        has_none = any([len(response["option"])==0 for response in responses])
        if has_none:
            print(f"Context:\n{context}")
            print()
            print(f"Question 1: {question1}")
            print(f"Answer 1: {answer1}")
            print()
            print(f"Question 2: {question2}")
            print(f"Answer 2: {answer2}")
            print()
            print(f"Choose the most appropriate option for each response: {options}")
            for response in responses:
                if not response["option"]:
                    print()
                    print(f"Response: {response['response']}")
                    while response["option"] not in ['A', 'B', 'C', 'D', 'E']:
                        option = input("Option (choose from A, B, C, D, E): ")
                        response["option"] = option.upper().strip()
                    
                    # save every input
                    with open(file_path, "w") as file:
                        json.dump(data, file, indent=4, ensure_ascii=False)

    # save every input
    with open(file, "r") as file:
        data = json.load(file)
