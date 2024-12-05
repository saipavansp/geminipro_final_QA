import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import json
from typing import List, Dict, Any
import time

# Streamlit page configuration
st.set_page_config(
    page_title="Document Chat & Quiz",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #f0f2f6;
    }
    .chat-question {
        font-weight: bold;
        color: #1f77b4;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        margin: 1rem 0;
    }
    .error-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        color: #721c24;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.docs_processed = False
    st.session_state.current_text = ""
    st.session_state.chat_history = []
    st.session_state.user_answers = []
    st.session_state.quiz_submitted = False


class DocumentProcessor:
    @staticmethod
    def get_pdf_text(pdf_docs: List[Any]) -> str:
        """Extract text from uploaded PDF documents."""
        try:
            text = ""
            for pdf in pdf_docs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")

    @staticmethod
    def get_text_chunks(text: str) -> List[str]:
        """Split text into chunks for processing."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000,
                chunk_overlap=1000,
                length_function=len
            )
            return text_splitter.split_text(text)
        except Exception as e:
            raise Exception(f"Error splitting text: {str(e)}")

    @staticmethod
    def get_vector_store(text_chunks: List[str]) -> None:
        """Create and save vector store from text chunks."""
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")


class ChatInterface:
    @staticmethod
    def get_conversational_chain():
        """Create the conversation chain for Q&A."""
        prompt_template = """
        Answer the question as detailed as possible from the provided context. 
        If the answer is not in the context, say "I don't find this information in the provided documents."
        Use bullet points for structured information.

        Context: {context}
        Question: {question}

        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)

    @staticmethod
    def process_user_input(user_question: str) -> str:
        """Process user questions and generate responses."""
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = vector_store.similarity_search(user_question)

            chain = ChatInterface.get_conversational_chain()
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

            answer = response["output_text"]

            # Add to chat history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": answer
            })

            return answer

        except Exception as e:
            raise Exception(f"Error processing question: {str(e)}")


class QuizGenerator:
    @staticmethod
    def clean_json_string(text: str) -> str:
        """Clean and validate JSON string from model response."""
        try:
            # Extract JSON content from code blocks if present
            if "```" in text:
                # Find the content between the first and last ```
                parts = text.split("```")
                for part in parts:
                    if "[" in part and "]" in part:
                        text = part
                        break

            # Remove any potential leading/trailing whitespace
            text = text.strip()

            # Basic string replacements for common issues
            replacements = {
                '\n': ' ',  # Remove newlines
                '\r': '',  # Remove carriage returns
                '}{': '},{',  # Fix missing commas between objects
                '"] [': '"], [',  # Fix array separators
                '] {': '], {',  # Fix missing commas after arrays
                '} [': '}, [',  # Fix missing commas before arrays
                '""': '"',  # Fix double quotes
                '\\"': '"',  # Fix escaped quotes
                '\\\\': '\\'  # Fix double escapes
            }

            for old, new in replacements.items():
                text = text.replace(old, new)

            # Ensure proper array wrapping
            text = text.strip()
            if not text.startswith('['):
                text = '[' + text
            if not text.endswith(']'):
                text = text + ']'

            # Remove multiple spaces
            text = ' '.join(text.split())

            return text
        except Exception as e:
            raise Exception(f"Error cleaning JSON string: {str(e)}")

    @staticmethod
    def generate_mcqs(text: str, num_questions: int = 10) -> List[Dict[str, Any]]:
        """Generate multiple choice questions from text."""
        try:
            model = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.3
            )

            prompt = f"""Generate exactly {num_questions} multiple choice questions based on the given text.
            Return ONLY a valid JSON array. Do not include any additional text, comments, or formatting.

            Required format:
            [
                {{
                    "question_text": "Simple question text here?",
                    "choices": ["Short option A", "Short option B", "Short option C", "Short option D"],
                    "correct_answer": 0,
                    "explanation": "Brief explanation here"
                }},
                ... (repeat for all questions)
            ]

            IMPORTANT RULES:
            1. Generate exactly {num_questions} questions
            2. Each question MUST have exactly 4 choices
            3. correct_answer MUST be 0, 1, 2, or 3
            4. Keep all text simple and avoid special characters
            5. Keep questions and answers concise
            6. Do not include any text outside the JSON array
            7. No markdown, no formatting, just plain JSON
            8. No nested objects or arrays except for the choices array

            Text for questions:
            {text[:3000]}"""  # Reduced text length for more reliable processing

            # Get response from model
            response = model.invoke(prompt)
            cleaned_response = QuizGenerator.clean_json_string(response.content)

            # First attempt to parse JSON
            try:
                mcqs = json.loads(cleaned_response)
            except json.JSONDecodeError:
                # If first attempt fails, try additional cleaning
                cleaned_response = cleaned_response.replace('\\', '')
                cleaned_response = cleaned_response.replace('"{', '{')
                cleaned_response = cleaned_response.replace('}"', '}')
                # Remove any trailing commas before closing brackets
                cleaned_response = cleaned_response.replace(',]', ']')
                cleaned_response = cleaned_response.replace(',}', '}')
                mcqs = json.loads(cleaned_response)

            # Validate and fix the MCQs
            validated_mcqs = []
            for mcq in mcqs:
                # Ensure all required keys exist
                if not all(key in mcq for key in ['question_text', 'choices', 'correct_answer', 'explanation']):
                    continue

                # Ensure exactly 4 choices
                if len(mcq['choices']) != 4:
                    continue

                # Ensure correct_answer is valid
                if not isinstance(mcq['correct_answer'], int) or mcq['correct_answer'] not in [0, 1, 2, 3]:
                    continue

                validated_mcqs.append(mcq)

            # If we don't have enough valid questions, raise an error
            if len(validated_mcqs) < num_questions:
                raise ValueError(
                    f"Could only generate {len(validated_mcqs)} valid questions out of {num_questions} requested")

            # Return exactly the number of questions requested
            return validated_mcqs[:num_questions]

        except Exception as e:
            raise Exception(f"Error generating quiz: {str(e)}")

def display_chat_message(message: Dict[str, str], index: int) -> None:
    """Display a single chat message with proper formatting."""
    with st.expander(f"Q: {message['question'][:50]}...", expanded=(index == 0)):
        st.markdown("**Question:**")
        st.write(message['question'])
        st.markdown("**Answer:**")
        st.write(message['answer'])


def display_quiz_results(mcqs: List[Dict[str, Any]], user_answers: List[int]) -> None:
    """Display quiz results with proper formatting."""
    correct_count = sum(1 for i, mcq in enumerate(mcqs)
                        if user_answers[i] == mcq['correct_answer'])

    st.subheader("📊 Quiz Results")
    st.write(f"Score: {correct_count}/{len(mcqs)} ({(correct_count / len(mcqs) * 100):.1f}%)")

    for i, (mcq, user_ans) in enumerate(zip(mcqs, user_answers)):
        with st.expander(f"Question {i + 1}: {mcq['question_text'][:50]}..."):
            st.write(f"**Your answer:** {mcq['choices'][user_ans]}")
            st.write(f"**Correct answer:** {mcq['choices'][mcq['correct_answer']]}")

            if user_ans == mcq['correct_answer']:
                st.success("✅ Correct!")
            else:
                st.error("❌ Incorrect")

            st.info(f"**Explanation:** {mcq['explanation']}")


def main():
    st.title("📚 Document Chat & Quiz Generator")

    # Initialize Google Gemini API
    if "GOOGLE_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    else:
        st.error("Please set your Google API key in Streamlit secrets.")
        return

    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat with Documents", "📝 Quiz Generator"])

    # Document Upload Sidebar
    with st.sidebar:
        st.header("📄 Document Upload")
        pdf_docs = st.file_uploader(
            "Upload your PDFs",
            accept_multiple_files=True,
            type=['pdf']
        )

        if st.button("Process Documents", type="primary"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
                return

            with st.spinner("Processing your documents..."):
                try:
                    processor = DocumentProcessor()
                    raw_text = processor.get_pdf_text(pdf_docs)
                    text_chunks = processor.get_text_chunks(raw_text)
                    processor.get_vector_store(text_chunks)

                    st.session_state.current_text = raw_text
                    st.session_state.docs_processed = True
                    st.success("✅ Documents processed successfully!")
                    time.sleep(1)  # Give user time to see success message
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return

    # Chat Tab
    with tab1:
        st.header("💭 Ask Questions About Your Documents")

        if not st.session_state.docs_processed:
            st.info("Please upload and process documents to start chatting.")
            return

        # Chat interface
        user_question = st.text_input("Your question:", key="user_question")
        if user_question:
            try:
                with st.spinner("Thinking..."):
                    response = ChatInterface.process_user_input(user_question)
                    if response:
                        st.markdown("### Answer:")
                        st.markdown(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                display_chat_message(chat, i)

    # Quiz Tab
    with tab2:
        st.header("📝 Generate Quiz")

        if not st.session_state.docs_processed:
            st.info("Please upload and process documents to generate a quiz.")
            return

        if 'mcqs' not in st.session_state and st.button("Generate New Quiz"):
            try:
                with st.spinner("Generating quiz questions..."):
                    quiz_gen = QuizGenerator()
                    st.session_state.mcqs = quiz_gen.generate_mcqs(st.session_state.current_text)
                    st.session_state.user_answers = [-1] * len(st.session_state.mcqs)
                    st.success("Quiz generated successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error generating quiz: {str(e)}")
                return

        if 'mcqs' in st.session_state:
            # Display quiz
            for i, mcq in enumerate(st.session_state.mcqs):
                st.subheader(f"Question {i + 1}")
                st.write(mcq['question_text'])

                choice = st.radio(
                    "Select your answer:",
                    mcq['choices'],
                    key=f"q_{i}",
                    index=st.session_state.user_answers[i] if st.session_state.user_answers[i] != -1 else None
                )

                if choice:
                    st.session_state.user_answers[i] = mcq['choices'].index(choice)

                st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Quiz", type="primary"):
                    if -1 in st.session_state.user_answers:
                        st.warning("Please answer all questions before submitting.")
                    else:
                        st.session_state.quiz_submitted = True
                        display_quiz_results(st.session_state.mcqs, st.session_state.user_answers)

            with col2:
                if st.button("Generate New Quiz", type="secondary"):
                    st.session_state.pop('mcqs', None)
                    st.session_state.pop('user_answers', None)
                    st.session_state.quiz_submitted = False
                    st.rerun()


if __name__ == "__main__":
    main()