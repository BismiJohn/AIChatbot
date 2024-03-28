import os
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Define the directory containing your text files
directory = "E:/Bismi/Chatbot_llama/Backend/Documents"

# Initialize a list to store the text content of each file
texts = []

# Iterate over each file in the directory
for filename in os.listdir(directory):
    # Ensure the file is a text file
    if filename.endswith(".txt"):
        # Construct the full path to the file
        filepath = os.path.join(directory, filename)
        
        # Read the content of the file and append it to the list
        with open(filepath, "r", encoding="utf-8") as file:
            texts.append(file.read())

# Load the language model
llm = CTransformers(model='E:/Bismi/Chatbot_llama/Backend/models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 256, 'temperature': 0.01})

# Increase the maximum context length
llm.config['max_context_length'] = 1024  # Set the maximum context length to 1024 tokens

# Load the interpreted information from the local database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})

# Create and save the local database
db = FAISS.from_texts(texts, embeddings)

# Prepare a version of the llm pre-loaded with the local content
retriever = db.as_retriever(search_kwargs={'k': 2})
prompt = PromptTemplate(
    template="""Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
""",
    input_variables=['context', 'question'])
qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type='stuff',
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': prompt})

# Loop for interacting with the AI via terminal
while True:
    try:
        # Ask the user for a question
        prompt = input("You: ")

        # Break the loop if the user enters 'exit'
        if prompt.lower() == 'exit':
            break

        # Get the AI's response
        output = qa_llm({'query': prompt})

        # Print the AI's response
        print("AI:", output["result"])

    except Exception as e:
        # Handle any unexpected errors gracefully
        print("An error occurred:", str(e))
        print("Please try again.")
