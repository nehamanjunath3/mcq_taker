import openai
from prompts import Prompter
from openai.error import OpenAIError
from utils import prepare_answer, get_api_key
import time


class HIPAgent:
    def __init__(self):
        """
        Initializes the HIPAgent instance with a new Prompter object.
        """

        self.prompter = Prompter()

    def get_response(self, question, answer_choices , prompt_type = "Zero-Shot", return_reasoning=False):
        """
        Calls the OpenAI 3.5 API to generate a response to the question.
        The response is then matched to one of the answer choices and the index of the
        matching answer choice is returned. If the response does not match any answer choice,
        -1 is returned.

        Args:
            prompt_type: The type of the prompt, either 'zero_shot', 'few_shot', or 'chain_of_thought'.
            row: The row containing the question and answer choices.

        Returns:
            Returns:
            int, str: The index of the answer choice that matches the response, and the reasoning (if return_reasoning is True).
            int: The index of the answer choice that matches the response (if return_reasoning is False).
        """
        # Set the OpenAI API key.
        openai.api_key = get_api_key()
        # openai.api_key = "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" # (own-key)
        # # openai.api_key = "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"

        # Generate the prompt.
        prompt = self.prompter.prepare_prompt(question, answer_choices, prompt_type)

        # Call the OpenAI 3.5 API.
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
        except OpenAIError as e:
            if e.http_status == 429:
                print("Rate limit exceeded. Waiting for 30 seconds...")
                time.sleep(30)
                print("Retrying now...")
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
            else:
                raise e

        response_text = response.choices[0].message.content

        
         # Prepare the answer.
        answer_index = prepare_answer(response_text)

        if return_reasoning:
            return answer_index, response_text

        else:
            return answer_index 
    
