import csv
from hip_agent import HIPAgent
from tqdm import tqdm
import pandas as pd


def get_pipeline(input_file, prompt_type):    # Parse the CSV file
    # with open(input_file, "r") as csvfile:
    #     reader = csv.reader(csvfile, delimiter=",")
    #     headers = next(reader)
    #     data = list(reader)

    data = pd.read_csv(input_file)

    # Get the correct answers
    correct_answers = []

    # Instantiate a HIP agent
    agent = HIPAgent()

    # Get the user's responses
    user_responses = []
    for index, row in tqdm(data.iterrows(), total=len(data))    :
        answer_choices = [row["answer_0"],row["answer_1"],row["answer_2"],row["answer_3"]]  
        correct_answers.append(answer_choices.index(row["correct"]))
        response = agent.get_response(row['question'], answer_choices, prompt_type)
        user_responses.append(response)

    # Calculate the score
    score = 0
    answers = []
    for i in range(len(data)):
        if user_responses[i] == correct_answers[i]:
            score += 1
            answers += [[1, user_responses[i], correct_answers[i]]]
        else:
            answers += [[0, user_responses[i], correct_answers[i]]]


    return score , answers 