# llama2-local-faiss-knowledge-chat-bot
# The Extreme Pro Max Ultra Plus Super local Chatbot

first create source_files folder, put your pdfs, txts, etc, supported file types:
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



then run ingest.py, this will convert all your files as a faiss vector db, for future embedding match

then run model_app.py, this will start the llama2 model enable to chat, you can select chat mode or not to chat with the vector db or not

query.py is not used, settings.py is for settings for embedding and model



# if you on windows, need to run these  
pip install git+https://github.com/huggingface/transformers.git   
python -m pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
