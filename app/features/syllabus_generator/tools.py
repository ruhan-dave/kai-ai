
    """
5 different prompts for 5 sections. Define each template by 
    """
class SyllabusBuilder:
    def __init__(self, num_questions, subject, pages, sections, user_type):
        self.num_questions = num_questions
        self.subject = subject
        self.pages = pages 
        self.sections = sections
        self.user_type = user_type # Are you a student or an educator? the user input will then be passed on to the llm
        self.system_template = """
            You are a syllabus generator on the subject: {subject} with total {pages} pages and {sections} sections.
            You then customize the content accoding to the {user_type}.

            Follow the instructions to create a syllabus:
            1. Generate a title based on the topic provided and context as key "title"
            2. Provide a welcome statement to the course and generate 2 sentences about the main objectives 
            of the course and what skills the class will gain as key "summary".
            3. Provide a few paragraphs for each section from the list of sections as key "content". 
            Clearly separate the "content" for each section and ensure relevance to each section.
            
            You must respond as a JSON object with the following structure:
            {{
                "subject": "<subject>",
                "summary": "<summary>",
                "learning_objectives": [
                    {{"key": "<section-1>", "value": "<content_1>"}},
                    {{"key": "<section-2>", "value": "<content>"}},
                    {{"key": "<section-3>", "value": "<content>"}},
                    {{"key": "<section>", "value": "<content>"}}
                ],
                "grading_policies": # ie: generate a dictionary of grades with ranges of scores represetned as percentages,
                "course_expections": # generate a paragraph that shows what the educator expects of the class, such as ethics, code of conduct, classroom policies, etc based on the pdf given by the user. ,
                "office_hours": # generate a JSON file:  {{"office_hours_insturctor": "<hours_instructors>", "office_hours_ta": "<hours_ta>"}}, where ...
                "schedule": # generate a dictionary of weeks to course content mapping, few sentences that describes course content and progress for that week
            }}

            For the "body" section of the JSON object above, You should continue to 
            generate "content" until the number of pages equal to {pages} the number of sections equal to {sections}.
            """
    
        def init_llm(self): """
        
        Task: Initialize the Large Language Model (LLM) for quiz question generation.

        Overview:
        This method prepares the LLM for generating quiz questions by configuring essential parameters such as the model name, temperature, and maximum output tokens. The LLM will be used later to generate quiz questions based on the provided topic and context retrieved from the vectorstore.

        Steps:
        1. Set the LLM's model name to "gemini-pro" 
        2. Configure the 'temperature' parameter to control the randomness of the output. A lower temperature results in more deterministic outputs.
        3. Specify 'max_output_tokens' to limit the length of the generated text.
        4. Initialize the LLM with the specified parameters to be ready for generating quiz questions.

        Implementation:
        - Use the VertexAI class to create an instance of the LLM with the specified configurations.
        - Assign the created LLM instance to the 'self.llm' attribute for later use in question generation.

        Note: Ensure you have appropriate access or API keys if required by the model or platform.
        """
        self.llm = VertexAI(
            ############# YOUR CODE HERE ############
            model_name="gemini-pro",
            temperature = 1.0,
            max_output_tokens = 50
        )
    def generate_gradings(self):

    def generate_course_expections(sef):

    def


    def generate_syllabus_with_vectorstore(self):
        """
        Task: Generate a quiz question using the topic provided and context from the vectorstore.

        Overview:
        This method leverages the vectorstore to retrieve relevant context for the quiz topic, then utilizes the LLM to generate a structured quiz question in JSON format. The process involves retrieving documents, creating a prompt, and invoking the LLM to generate a question.

        Prerequisites:
        - Ensure the LLM has been initialized using 'init_llm'.
        - A vectorstore must be provided and accessible via 'self.vectorstore'.

        Steps:
        1. Verify the LLM and vectorstore are initialized and available.
        2. Retrieve relevant documents or context for the quiz topic from the vectorstore.
        3. Format the retrieved context and the quiz topic into a structured prompt using the system template.
        4. Invoke the LLM with the formatted prompt to generate a quiz question.
        5. Return the generated question in the specified JSON structure.

        Implementation:
        - Utilize 'RunnableParallel' and 'RunnablePassthrough' to create a chain that integrates document retrieval and topic processing.
        - Format the system template with the topic and retrieved context to create a comprehensive prompt for the LLM.
        - Use the LLM to generate a quiz question based on the prompt and return the structured response.

        Note: Handle cases where the vectorstore is not provided by raising a ValueError.
        """
        ############# YOUR CODE HERE ############
        # Initialize the LLM from the 'init_llm' method if not already initialized
        self.init_llm()
        # Raise an error if the vectorstore is not initialized on the class
        if not self.vectorstore:
            raise ValueError("The vectorstore is not initialized")
        ############# YOUR CODE HERE ############
        
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel

        ############# YOUR CODE HERE ############
        # Enable a Retriever using the as_retriever() method on the VectorStore object
        # HINT: Use the vectorstore as the retriever initialized on the class
        ############# YOUR CODE HERE ############
        retriever = self.vectorstore.db.as_retriever()
        
        ############# YOUR CODE HERE ############
        # Use the system template to create a PromptTemplate
        # HINT: Use the .from_template method on the PromptTemplate class and pass in the system template
        ############# YOUR CODE HERE ############
        
        prompt_template = PromptTemplate.from_template(self.system_template)

        # RunnableParallel allows Retriever to get relevant documents
        # RunnablePassthrough allows chain.invoke to send self.topic to LLM
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "topic": RunnablePassthrough()}
        )
        
        ############# YOUR CODE HERE ############
        # Create a chain with the Retriever, PromptTemplate, and LLM
        # HINT: chain = RETRIEVER | PROMPT | LLM 
        ############# YOUR CODE HERE ############
        # think the 1st argument should be setup_and_retrieval, not just retriever
        chain = setup_and_retrieval | prompt_template | self.llm
        # Invoke the chain with the topic as input
        response = chain.invoke(self.topic)
        return response

class RAGpipeline:
    def __init__(self, query, vectordb, verbose=False):
        self.vectordb = vectordb

class Vectordb:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        