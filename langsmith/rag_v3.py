import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

load_dotenv()  # expects OPENAI_API_KEY in .env
os.environ['LANGCHAIN_PROJECT'] = 'RAG'
PDF_PATH = "langsmith\islr.pdf"  

# 1) Load PDF
@traceable(name='load-pdf')
def load_pdf(path: str):
    loader = PyPDFLoader(PDF_PATH)
    return loader.load()

@traceable(name='splitter')
def split_documents(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

# 3) Embed + index
@traceable(name='vector-stor')
def store_embeddings(splits):
    emb = MistralAIEmbeddings(
        model="mistral-embed",
        api_key="JgkR1k8M86MlnwMOe89NZxmfgMaVinWo")
    vs = FAISS.from_documents(splits, emb)
    return vs

@traceable(name="setup-retriever")
def retriever_pipeline(pdf_path: str, chunk_size=1000, chunk_overlap=150):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size, chunk_overlap)
    vs = store_embeddings(splits)

    return vs


# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatAnthropic(model='claude-3-5-sonnet-20241022')
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

@traceable(name="retrival-to-rag")
def rag_pipeline(pdf_path: str, question: str):
    vectorstore = retriever_pipeline(pdf_path, chunk_size=1000, chunk_overlap=150)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
    })

    chain = parallel | prompt | llm | StrOutputParser()

    lc_config = {"run_name": "pdf_rag_query"}
    return chain.invoke(question, config=lc_config)

if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = rag_pipeline(PDF_PATH, q)
    print("\nA:", ans)
