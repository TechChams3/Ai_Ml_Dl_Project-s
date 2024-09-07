import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()


def get_vectorStore_from_url(url):
   loader = WebBaseLoader(url)
   documents = loader.load()

   text_spliter = RecursiveCharacterTextSplitter()
   document_chank = text_spliter.split_documents(documents)
   vector_store = Chroma.from_documents(document_chank , OpenAIEmbeddings())
   return vector_store



def get_Context_retriever_chain(vector_store):
   
   llm = ChatOpenAI()
   retriever = vector_store.as_retriever()
   prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name = "chat_history"),
      ("user","{input}"),
      ("user","lorem")
   ])

   retriver_chain = create_history_aware_retriever(llm,retriever,prompt)
   return retriver_chain



def get_conversational_reg_chain(retriever_chain):
   llm = ChatOpenAI()

   prompt = ChatPromptTemplate.from_messages([
      ("system","Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name = 'chat_history'),
      {"user","{input}"},
   ])

   stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
   return create_retrieval_chain(retriever_chain,stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_Context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_reg_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']










st.set_page_config(page_title="Chat With Website",page_icon="XX")
st.title("Chat with websites")
if "chat_history" not in st.session_state:
   st.session_state.chat_history = [
    AIMessage(content="Hellow, I am a bot. How can I help you")
]


with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website Url")



if website_url is None or website_url == "":
   st.info("please enter a website URL")

else:
    if "chat_history" not in st.session_state:
       st.session_state.chat_history = [
          AIMessage(content="Hello, iam anot")
       ]

    if "vector_store" not in st.session_state:
       st.session_state.vector_store = get_vectorStore_from_url(website_url)

      



    user_query = st.chat_input("Type your message hear>>>")
    if user_query is not None and user_query != "":
     response = get_response(user_query)
     st.session_state.chat_history.append(HumanMessage(content=user_query))
     st.session_state.chat_history.append(AIMessage(content=response))
  
    for message in st.session_state.chat_history:
     if isinstance(message,AIMessage):
      with st.chat_message("AI"):
         st.write(message.content)
     elif isinstance(message,HumanMessage):
      with st.chat_message("Human"):
         st.write(message.content)





