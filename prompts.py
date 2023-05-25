import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
from utils import prepare_document, calculate_similarity, read_embeddings, get_context
import tiktoken
import numpy as np
import traceback


class Prompter:
    def __init__(self):
        """
        Initializes the Prompter instance.
        """
        
        self.base_prompt = f"""Act as if you are taking the Medical entrance exam. Please answer the following question in the exact format mentioned below. \n\n: 
        Correct_Answer: <LETTER> \n """

        self.reasoning = f"""Explanation of the correct answer: <EXPLANATION>
        Explanation of the incorrect answers: <EXPLANATION>"""

        self.example_questions =  f""" Here are a few examples of the MCQ format. \n\n
        What is the capital of France \n(a) Paris \n(b) Barcelona \n(c) Spain \n(d) Antarctica \n\n Correct Answer: (a) \n\n 
        Footballer Messi is from which country: \n(a) Brazil \n(b) Portugal \n(c) Argentina \n(d) Sudan \n\n Correct Answer: (c) \n\n
        Which of these are prime numbers: \n(a) 2 \n(b) 3 \n(c) 5 \n(d) all of the above \n Correct Answer: (d)\n
        Which of these cities are in India: \n(a) Delhi \n(b) London \n(c) Mumbai \n(d) both a and c \nCorrect Answer: (d)\n"""



    def zero_shot_prompt(self ,question_text, options):
        """Generate a zero-shot learning prompt."""
        prompt = f"{self.base_prompt}\n \nQuestion: {question_text}\n" \
                  f"(a) {options[0]}\n(b) {options[1]}\n(c) {options[2]}\n(d) {options[3]} \nAnswer: ".strip()
        
        return prompt

    def few_shot_prompt(self, question_text, options):
        """
        Generates a zero-shot learning prompt.

        Args:
            question_text (str): The question text.
            options (list): The list of answer options.

        Returns:
            str: The generated prompt.
        """        
        prompt = f"{self.base_prompt}\n \n{self.example_questions}\n \n" \
                  f"Question: {question_text}\n (a) {options[0]}\n(b) {options[1]}\n(c) {options[2]}\n(d) {options[3]} \nAnswer: ".strip()
        
        return prompt

    def chain_of_thought_prompt(self, question_text, options):
        """
        Generates a few-shot learning prompt.

        Args:
            question_text (str): The question text.
            options (list): The list of answer options.

        Returns:
            str: The generated prompt.
        """
        question_text, options = question_text, options
        prompt = f"""{self.base_prompt}\n \n{self.reasoning} \n Let’s think step by step. 
        Question: {question_text}
        {question_text}\n(a) {options[0]}\n(b) {options[1]}\n(c) {options[2]}\n(d) {options[3]}
        For each option, determine whether it is true or false. If it is false, explain why it is false. If it is true, explain why it is true.:
        In light of the above explanations, which of the following is the correct answer? 
        Answer:""".strip()
        return prompt
    
    def RAG_prompt(self, question_text, options, document=None):
        """
        Generates a retrieval augmented generation prompt.
        Args:
            question_text (str): The question text.
            options (list): The list of answer options.
            document (str): The document to be used for retrieval.

        Returns:
            str: The generated prompt.

        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            embedding_model = "text-embedding-ada-002"
            tokens_limit = 4096
            embedding_model = "text-embedding-ada-002"
            df = prepare_document(document) if document else read_embeddings('textbook_embeddings.csv')
            quest_embedding = get_embedding(question_text , engine = embedding_model)

            df["similarity"] = calculate_similarity(df, quest_embedding)

            results = (df.sort_values("similarity", ascending=False))

            user_start = self.base_prompt + self.reasoning + "Answer the question based on the context below.\n\n"+ "Context:\n"
            user_end =  (
            f"""\n\nQuestion: {question_text}\n
            
            (a) {options[0].strip()}
            (b) {options[1].strip()}
            (c) {options[2].strip()}
            (d) {options[3].strip()}
            Answer:"""
        )
            
            count_of_tokens_consumed = len(encoding.encode("\"role\":\"user\"" + ", \"content\" :\""+user_start + "\n\n---\n\n" + user_end))

            count_of_tokens_for_context = tokens_limit - count_of_tokens_consumed

            contexts = get_context(results, count_of_tokens_for_context)

            complete_prompt =  user_start + contexts + "\n\n---\n\n" + user_end


        except Exception as e:
            print("An error occurred while generating the RAG prompt. Details:")
            print(traceback.format_exc())

        return complete_prompt
    

    def prepare_prompt(self, question, choices, prompt_type):
        """Prepare the final prompt based on the prompt type
        Args:
            question (str): The question text.
            choices (list): The list of answer options.
            prompt_type (str): The type of prompt to be generated.
        Returns:
            str: The generated prompt.
            """

        question = question[(question.find(". ") + 1) :].strip()
        choices = [choices[i].strip() for i in range(len(choices))]


        if prompt_type == "Zero-Shot":
            return self.zero_shot_prompt(question, choices)
        elif prompt_type == "Few-Shot":
            return self.few_shot_prompt(question, choices)
        elif prompt_type == "Chain of thought":
            return self.chain_of_thought_prompt(question, choices)
        elif prompt_type == "Retrieval Augmented Generation":
            return self.RAG_prompt(question, choices)
        else:
            raise ValueError("Invalid prompt type. Must be either 'Zero-Shot', 'Few-Shot', 'Chain of thought' or 'Retrieval Augmented Generation'")
