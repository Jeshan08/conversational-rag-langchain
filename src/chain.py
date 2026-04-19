from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()

def get_chain(retriever):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    template = """Answer the question based only on the following context:
    {context}

    Chat History :
    {chat_history}

    Question: {question}
    Answer: """

    prompt = ChatPromptTemplate.from_template(template)

    def docs2str(docs):
      return "\n\n".join(doc.page_content for doc in docs)

    def format_history(chat_history):
      formatted = ""
      for message in chat_history:
        if isinstance(message, HumanMessage):
          formatted += f"Human: {message.content}\n"
        elif isinstance(message, AIMessage):
          formatted += f"AI: {message.content}\n"
      return formatted
    

    rag_chain = (
      {
        "context": lambda x: docs2str(retriever.invoke(x["question"]+ " " + format_history(x["chat_history"]))),
        "question": lambda x: x["question"],
        "chat_history": lambda x: format_history(x["chat_history"])
      }
      | prompt
      | llm
      | StrOutputParser()
    )

    return rag_chain