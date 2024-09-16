from app.services.logger import setup_logger 
from app.features.syllabus_generator.tools import RAGpipeline, SyllabusBuilder
from app.api.error_utilities import LoaderError, ToolExecutorError

logger = setup_logger()

def executor(files: list, topic: str, num_questions: int, verbose=False):
    try:
        if verbose:
            logger.debug(f"Files: {files}")

            rag = RAGpipeline(verbose=verbose)
            rag.compile()

            # Create and return the quiz questions
            output = SyllabusBuilder(db, topic, verbose=verbose).create_questions(num_questions, difficulty)

    except LoaderError as e:
        error_message = e
        logger.error(f"Error in RAGPipeline -> {error_message}")
        raise ToolExecutorError(error_message)
    
    except Exception as e:
        error_message = f"Error in executor: {e}"
        logger.error(error_message)
        raise ValueError(error_message)
    
    return output

