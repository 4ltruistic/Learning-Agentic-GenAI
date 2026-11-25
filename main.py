import os
os.environ["USER_AGENT"] = "my-rag-learning-app/1.0"
from langchain_community.document_loaders import  PyPDFDirectoryLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.output_parsers import  StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from dotenv import load_dotenv

load_dotenv()

chat_history=[]

context=[]

def docs_loader_list(*loaded_doc):
    combined=[]
    for docs in loaded_doc:
        combined.extend(docs)
    return combined


PERSIST_DIR="./chroma_db"
Collection_name="research_papers"
# Embeddings function
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Generative model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3
)
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3
)

vector_db=Chroma(
        collection_name=Collection_name,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIR
    )
existing_data = vector_db.get()

# Embeddings generation 

if len(existing_data["ids"]) > 0:
    print("Fetching the embeddings locally!")

else:
    print("Creating the embeddings!")

    pdf_loader=PyPDFDirectoryLoader(
        path=r"D:\Research papers (DL)"
    )

    web_laoder=WebBaseLoader(
        web_path=r"https://en.wikipedia.org/wiki/Attention_Is_All_You_Need"
    )

    web_docs=web_laoder.load()
    pdf_docs=pdf_loader.load()

    docs_combined=docs_loader_list(web_docs,pdf_docs)

    # Text Splitter

    Text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splitted_doc=Text_splitter.split_documents(docs_combined)

    # vector store
    vector_db=Chroma.from_documents(
        documents=splitted_doc,
        embedding=embedding_model,
        collection_name=Collection_name,
        persist_directory="./chroma_db"
    )


# retreiver
Retreiver=vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={'k':3}
)


# for i, doc in enumerate(vecs):
#     context.append(doc.page_content)

# Template1=PromptTemplate(
#     template="Give the most accurate match of {user_input}",
#     input_variables=['user_input']
# )

Template2=PromptTemplate(
    template="Give the answer using the chat history {chat_history} and the context {context}",
    input_variables=['chat_history','context']
)

parser=StrOutputParser()

chain2=Template2 | model | parser

while True:
    user_input=input("YOU: ")
    if(user_input=="exit"):
        break
    chat_history.append(
        HumanMessage(content=user_input)
    )
    # print("=="*20)
    print(user_input)
    context_respone=Retreiver.invoke(user_input)
    for i, doc in enumerate(context_respone):
        context.append(doc.page_content)

    response= chain2.invoke({"chat_history":chat_history,"context":context})
    print(response)
    chat_history.append(
        AIMessage(content=response)
    )
    # print("-"*20)
    # for i in chat_history:
    #     print(i)
    #     print("-"*20)
    
    # print("+"*20)





