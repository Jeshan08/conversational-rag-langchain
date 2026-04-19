from loader import load_documents
from retriever import get_retriever
from chain import get_chain

from langchain_core.messages import HumanMessage, AIMessage

def main():
  print("Loading documents")
  chunks = load_documents()

  print("Creating retriever")
  retriever = get_retriever(chunks)

  print("Creating Rag chain")
  rag_chain = get_chain(retriever)

  # empty as in the begin no history
  chat_history = []

  print("\n RAG System Ready! Ask your questions!")
  print("Type 'exit' to quit\n")


  while True:
    question = input("You: ")
    
    if question.lower() == "exit":
      print("Goodbye!")
      break

    answer = rag_chain.invoke(
      {
        "input": question,
        "chat_history" : chat_history
      }
    )
    # appending history with every question
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer['answer']))

    print(f"Answer: {answer['answer']}\n")

if __name__ == "__main__":
  main()