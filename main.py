import os
os.environ["USER_AGENT"] = "my-rag-learning-app/1.0"
from langchain_community.document_loaders import  PyPDFDirectoryLoader, WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.output_parsers import  StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate,MessagesPlaceholder
from dotenv import load_dotenv
from tkinter import Tk, filedialog
from configs import (
    embedding_model,chat_history,model, vector_db,context, Collection_name
)



load_dotenv()


def file_selection():
    # Open Windows file picker
    root = Tk()
    root.withdraw()

    selected_files = filedialog.askopenfilenames(
        title="Select Research Papers",
        filetypes=[
            ("PDF files", "*.pdf")
        ]
    )
    if not selected_files:
        print("no files selected")
    
    root.destroy()
    return selected_files



def generate_embeddings():
    pdf_docs = []
    selected_files=file_selection()
    if not selected_files:
        print("No files Selected!")
        return
    for file_path in selected_files:
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
    
            pdf_docs.extend(loaded_docs)

    Text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splitted_doc=Text_splitter.split_documents(pdf_docs)
    existing_data=vector_db.get()
    if(len(existing_data["ids"])>0):
        print("Adding Embeddings in DB!")
        vector_db.add_documents(
            documents=splitted_doc
        )
        return vector_db
    else:
        print("Making the DB!")
            # vector store
        temp_db=Chroma.from_documents(
            documents=splitted_doc,
            embedding=embedding_model,
            collection_name=Collection_name,
            persist_directory="./chroma_db"
        )
        return temp_db


    


def docs_loader_list(*loaded_doc):
    combined=[]
    for docs in loaded_doc:
        combined.extend(docs)
    return combined


# retreiver
Retreiver=vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={'k':3}
)

prompt=ChatPromptTemplate.from_messages([
    ("system","""You are a helpful research assistant.
                    Answer the user's question using the retrieved context.
                    Use chat history only to understand the conversation.
                    If the answer is not present in the context, clearly say
                    that you could not find the answer in the provided documents.
                    Give the answer using the chat history,context and user_input
                    Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{user_input}")    

]
    
)

parser=StrOutputParser()

chain2=prompt | model | parser

while True:
    user_input=input("YOU: ")
    if(user_input=="exit"):
        break

    if(user_input=="add"):
        vector_db= generate_embeddings()
        continue


    chat_history.append(
        HumanMessage(content=user_input)
    )
    if(len(chat_history)>10):
        chat_history=chat_history[-10:]


    print("=="*20)
    # print(f"YOU: {user_input}")
    context_respone=Retreiver.invoke(user_input)
    # for i, doc in enumerate(context_respone):
    #     context.append(doc.page_content)
    
    context="\n\n".join(
        doc.page_content for doc in context_respone
    )
    

    response= chain2.invoke({"chat_history":chat_history,"context":context,"user_input":user_input})
    print(f"AI: {response}")
    print("=="*20)
    chat_history.append(
        AIMessage(content=response)
    )
    chat_history.append(
        HumanMessage(content=user_input)
    )
    # context.clear()
    # print("-"*20)
    # for i in chat_history:
    #     print(i)
    #     print("-"*20)
    
    # print("+"*20)





