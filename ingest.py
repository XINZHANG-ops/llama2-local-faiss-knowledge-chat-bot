import logging
import os
from tqdm import tqdm
from test_vector_dbs.faissDB.faissDB import faissDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import hash_string
from collections import defaultdict
from settings import (chunk_size,
                      chunk_overlap,
                      source_dir,
                      embedding_function,
                      document_loader_map,
                      index_name,
                      index_type,
                      metric,
                      dimension
                      )
logging.basicConfig(level=logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
processed_track_file = 'processed_files.txt'


def find_new_and_deleted_docs(source_dir, processed_files_path):
    all_files = set(os.listdir(source_dir))

    # 加载已经处理过的文件名
    try:
        with open(processed_files_path, 'r') as f:
            processed_files = set(f.read().splitlines())
    except FileNotFoundError:
        processed_files = set()

    # 找出新文件和被删除的文件
    new_files = all_files - processed_files
    deleted_files = processed_files - all_files

    # 更新已处理的文件列表
    processed_files = processed_files - deleted_files
    return new_files, deleted_files, processed_files


new_files, deleted_files, processed_files = find_new_and_deleted_docs(source_dir, processed_track_file)


def load_single_documents(file_path):
    file_extension = os.path.splitext(file_path)[1]
    loader_class = document_loader_map.get(file_extension)
    if loader_class:
        file_loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return file_loader.load()[0]


def load_multiple_documents(source_dir, new_files_names, processed_files_names, processed_files_path):
    documents = []
    # 处理新文件
    for file_name in tqdm(new_files_names):
        file_path = os.path.join(source_dir, file_name)
        single_document = load_single_documents(file_path)
        documents.append(single_document)
        processed_files_names.add(file_name)  # 添加到已处理文件列表

    # 保存更新后的已处理文件列表
    with open(processed_files_path, 'w') as f:
        for file_name in processed_files_names:
            f.write(f"{file_name}\n")
    return documents


if new_files:
    # load the document and split it into chunks
    documents = load_multiple_documents(source_dir, new_files, processed_files, processed_track_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)

    # convert doc conetents into embeddings and ids
    hash_collision_record = defaultdict(list)
    content2id = dict()
    id2content = dict()

    for doc in documents:
        content = doc.page_content
        content_hash = hash_string(content)
        hash_collision_record[content_hash].append(content)
        content2id[content] = content_hash
        id2content[content_hash] = content

    for hid, cs in hash_collision_record.items():
        if len(cs) >= 2:
            logging.info(f"Hash collision for id {hid}, with {len(cs)} docs!")
        else:
            pass
    # // TODO 当删除了文件时候，可想方法删除db里面相应的vector
    # check if the index folder exist
    if os.path.exists(index_name):
        existed_file_ids = []
        new_added_file_ids = []
        db = faissDB.load(index_name)
        logging.info(f"Loaded index {index_name}.")
        for doc_content, hid in tqdm(content2id.items()):
            if hid in db.id2content:
                existed_file_ids.append(hid)
            else:
                new_added_file_ids.append(hid)
                db.add([embedding_function(doc_content)], [hid])
                db.hash_collision_record[hid].append(doc_content)
                db.id2content[hid] = doc_content
                db.content2id[doc_content] = hid
        logging.info(f"{len(existed_file_ids)} chunks exist, passed.")
        logging.info(f"{len(new_added_file_ids)} new chunks added.")
    else:
        new_added_file_ids = []
        db = faissDB(index_type, metric=metric, d=dimension, index_name=index_name)
        logging.info(f"Created new index {index_name}.")
        for doc_content, hid in tqdm(content2id.items()):
            new_added_file_ids.append(hid)
            db.add([embedding_function(doc_content)], [hid])
        db.hash_collision_record = hash_collision_record
        db.id2content = id2content
        db.content2id = content2id
        logging.info(f"{len(new_added_file_ids)} new chunks added.")
    db.save()
else:
    # 保存更新后的已处理文件列表
    with open(processed_track_file, 'w') as f:
        for file_name in processed_files:
            f.write(f"{file_name}\n")
    logging.info(f"{len(processed_files)} files exist, no new file added, passed.")
