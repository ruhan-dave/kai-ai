from typing import List, Tuple, Dict, Any
from io import BytesIO
from fastapi import UploadFile
from pypdf import PdfReader
from urllib.parse import urlparse
import requests
import os
import json
import time

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.services.logger import setup_logger
from app.services.tool_registry import ToolFile
from app.api.error_utilities import LoaderError

logger = setup_logger(__name__)

def transform_json_dict(input_data: dict) -> dict:
    # Validate and parse the input data to ensure it matches the QuizQuestion schema
    syllabus = Syllabus(**input_data)

    # Transform the choices list into a dictionary
    transformed_grading_policies = {grading_policies.key: grading_policies.value for grading_policies in syllabus.grading_policies}
    transformed_schedule = {schedule.key: schedule.value for schedule in syllabus.schedule}

    # Create the transformed structure
    transformed_data = {
        "subject": syllabus.subject,
        "summary": syllabus.summary,
        "learning_objectives": syllabus.learning_objectives,
        "grading_policies": transformed_grading_policies,
        "course_expectations": syllabus.course_expectations,
        "office_hours": syllabus.office_hours,
        "schedule": transformed_schedule
    }

    return transformed_data

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)
    
    with open(absolute_file_path, 'r') as file:
        return file.read()

class SyllabusBuilder:
    def __init__(self, vectorstore, subject, prompt=None, model=None, parser=None, verbose=False):
        default_config = {
            "model": GoogleGenerativeAI(model="gemini-1.0-pro"),
            "parser": JsonOutputParser(pydantic_object=Syllabus),
            "prompt": read_text_file("prompt/syllabus-prompt.txt")
        }

        self.prompt = prompt or default_config["prompt"]
        self.model = model or default_config["model"]
        self.parser = parser or default_config["parser"]
        
        self.vectorstore = vectorstore
        self.subject = subject
        self.verbose = verbose
        
        if vectorstore is None: raise ValueError("Vectorstore must be provided")
        if subject is None: raise ValueError("Subject must be provided")
    
    def compile(self):
        # Return the chain
        prompt = PromptTemplate(
            template=self.prompt,
            input_variables=["subject"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        retriever = self.vectorstore.as_retriever()
        
        runner = RunnableParallel(
            {"context": retriever, "subject": RunnablePassthrough()}
        )
        
        chain = runner | prompt | self.model | self.parser
        
        if self.verbose: logger.info(f"Chain compilation complete")
        
        return chain
    
    def validate_response(self, response: Dict) -> bool:
        try:
            # Assuming the response is already a dictionary
            if isinstance(response, dict):
                if 'subject' in response and 'summary' in response and 'learning_objectives' in response and 'grading_policies' in response and 'course_expectations' in response and 'office_hours' in response and 'schedule' in response:
                    grading_policies = response['grading_policies']
                    schedule = response['schedule']
                    if isinstance(grading_policies, dict) and isinstance(schedule, dict):
                        for key, value in grading_policies.items():
                            if not isinstance(key, str) or not isinstance(value, str):
                                return False
                        for key, value in schedule.items():
                            if not isinstance(key, str) or not isinstance(value, str):
                                return False
                        return True
            return False
        except TypeError as e:
            if self.verbose:
                logger.error(f"TypeError during response validation: {e}")
            return False

    def format_grading_policies(self, grading_policies: Dict[str, str]) -> List[Dict[str, str]]:
        return [{"key": k, "value": v} for k, v in grading_policies.items()]
    
    def format_schedule(self, schedule: Dict[str, str]) -> List[Dict[str, str]]:
        return [{"key": k, "value": v} for k, v in schedule.items()]

    def create_questions(self) -> List[Dict]:
        if self.verbose: logger.info(f"Creating syllabus")
        
        chain = self.compile()

        attempts = 0
        max_attempts = 5  # Allow for more attempts to generate questions

        while attempts < max_attempts:
            response = chain.invoke(self.subject)
            if self.verbose:
                logger.info(f"Generated response attempt {attempts + 1}: {response}")

            response = transform_json_dict(response)
            # Directly check if the response format is valid
            if self.validate_response(response):
                response["grading_policies"] = self.format_grading_policies(response["grading_policies"])
                response["schedule"] = self.format_schedule(response["schedule"])
                if self.verbose:
                    logger.info(f"Valid syllabus generated: {response}")
            else:
                if self.verbose:
                    logger.warning(f"Invalid response format. Attempt {attempts + 1} of {max_attempts}")
            
            # Move to the next attempt regardless of success to ensure progress
            attempts += 1
        
        if self.verbose: logger.info(f"Deleting vectorstore")
        self.vectorstore.delete_collection()
        
        # Return the list of questions
        return response

class Syllabus(BaseModel):
    subject: str = Field(description="The subject of the course for which the syllabus is designed")
    summary: str = Field(description="A summary of the generated syllabus")
    learning_objectives: str = Field(description="Main objectives of the course and skills students will gain from the course")
    grading_policies: Dict[str, str] = Field(description="a dictionary with range of numerical grades as keys and letter grades as values")
    course_expectations: str = Field("A paragraph that shows what the educator expects of the class, such as ethics, code of conduct, classroom policies, etc., based on the PDF file given by the user")
    office_hours: str = Field("Office hours of the instructor and teaching assistants")
    schedule: Dict[str, str] = Field(description="A dictionary with each week of course as keys and respective weekly course contents as values")