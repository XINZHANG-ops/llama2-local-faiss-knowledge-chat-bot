import os
from sentence_transformers import SentenceTransformer
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings


##################################
# faiss db settings
##################################
def embedding_function(content):
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embedding = [float(i) for i in list(embedding_model.encode(content, convert_to_tensor=False, show_progress_bar=False))]
    return embedding


index_name = "paper_embeddings"
index_type = "0"
metric = "cosine"
dimension = len(embedding_function("test"))


##################################
# split settings
##################################
chunk_size = 1000
chunk_overlap = 500

source_folder = "source_files"

root_dir = os.path.dirname(os.path.realpath(__file__))
source_dir = os.path.join(root_dir, source_folder)

document_loader_map = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}


##################################
# search settings
##################################
top_k_match = 5

temperature = 0.1
max_length = 1024
top_k = 50
top_p = 0.9
new_chat = False
chat_mode = True
embedding_hist = 0
system_message = "You are helpful assistant, answer whatever I ask."


def create_context(similar_docs):
    context_template = ""
    for idx, doc_text in enumerate(similar_docs):
        context_template += f"Context {idx+1}:\n{doc_text}\n\n"
    return context_template


prompt_template = """Use the following pieces of context to answer the question at the end, try to use the most relevant context. If you don't know the answer,\
just say that you don't know, don't try to make up an answer.

{context}
Question: {question}
Helpful Answer:"""
