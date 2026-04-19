from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


from dotenv import load_dotenv

load_dotenv()

def get_chain(retriever):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
      ("system", """Given the chat history and latest user question,
      rephrase the question to be standalone and clear.
      Don't answer it, just rephrase it.
      If it's already clear, return as is."""),
      MessagesPlaceholder(variable_name="chat_history"),
      ("human", "{input}")
    ])
#  smart retriever who first gets the rephrases message before sending to retirver 
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
        Answer using only the context below.
        If you don't know, say you don't know.
        Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # def docs2str(docs):
    #   return "\n\n".join(doc.page_content for doc in docs)

    # def format_history(chat_history):
    #   formatted = ""
    #   for message in chat_history:
    #     if isinstance(message, HumanMessage):
    #       formatted += f"Human: {message.content}\n"
    #     elif isinstance(message, AIMessage):
    #       formatted += f"AI: {message.content}\n"
    #   return formatted
    

    # rag_chain = (
    #   {
    #     "context": lambda x: docs2str(retriever.invoke(x["question"]+ " " + format_history(x["chat_history"]))),
    #     "question": lambda x: x["question"],
    #     "chat_history": lambda x: format_history(x["chat_history"])
    #   }
    #   | prompt
    #   | llm
    #   | StrOutputParser()
    # )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    return rag_chain