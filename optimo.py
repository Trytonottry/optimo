import os
import subprocess
from typing import List, Dict
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_callback_manager

def load_and_preprocess_documents(
  file_paths: List[str], text_splitter: RecursiveCharacterTextSplitter,
) -> List[Dict]:
  """
  Загружает документы и разбивает текст на куски.
  """
  documents = []
  for file_path in file_paths:
    loader = UnstructuredFileLoader(file_path)
    loaded_docs = loader.load()
    for doc in loaded_docs:
      chunks = text_splitter.split_text(doc.page_content)
      for chunk in chunks:
        documents.append({"text": chunk, "metadata": {"source": file_path}})
  return documents

def create_vector_store(documents: List[Dict], embedding_function) -> FAISS:
  """
  Создает векторный магазин с использованием FAISS.
  """
  return FAISS.from_documents(documents, embedding_function)

def create_retrieval_qa_chain(vectorstore: FAISS, llm: OpenAI) -> RetrievalQA:
  """
  Создает цепочку поиска ответов с использованием RAG.
  """
  retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
  return RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever
  )

def check_code_compliance(code: str, qa_chain: RetrievalQA) -> str:
  """
  Проверяет код на соответствие правилам и стандартам.
  """
  query = f"Проверьте этот код на соответствие правилам и стандартам: {code}"
  return qa_chain.run(query)

def run_linters(file_path: str, linter_commands: List[str]) -> List[str]:
  """
  Запускает линтеры и возвращает результаты.
  """
  results = []
  for command in linter_commands:
    try:
      output = subprocess.check_output(command.split(), stderr=subprocess.STDOUT).decode("utf-8")
      results.append(output)
    except subprocess.CalledProcessError as e:
      results.append(f"Ошибка при запуске линтера: {e}")
  return results

def main():
  """
  Основной код для инструмента проверки кода.
  """
  # Загрузка правил и стандартов компании
  file_paths = ["company_standards.txt", "coding_guidelines.md"]
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  documents = load_and_preprocess_documents(file_paths, text_splitter)

  # Создание векторного магазина
  embeddings = OpenAIEmbeddings()
  vectorstore = create_vector_store(documents, embeddings)

  # Создание цепочки поиска ответов
  llm = OpenAI(temperature=0.7)
  qa_chain = create_retrieval_qa_chain(vectorstore, llm)

  # Проверка проекта
  project_path = "/path/to/project"
  file_paths = [os.path.join(project_path, f) for f in os.listdir(project_path) if os.path.isfile(os.path.join(project_path, f))]
  for file_path in file_paths:
    # Запуск линтеров
    linter_commands = ["flake8 --max-line-length=120", "pylint"]
    linter_results = run_linters(file_path, linter_commands)

    # Проверка кода с помощью RAG
    with open(file_path, "r") as f:
      code = f.read()
    compliance_result = check_code_compliance(code, qa_chain)

    # Вывод результатов
    print(f"Файл: {file_path}")
    print("Результаты линтеров:")
    for result in linter_results:
      print(result)
    print("Результат проверки на соответствие стандартам:")
    print(compliance_result)

if __name__ == "__main__":
  main()
