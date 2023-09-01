import requests
import logging
import textwrap
from test_vector_dbs.faissDB.faissDB import faissDB
from settings import (index_name,
                      embedding_function,
                      top_k_match,
                      system_message,
                      create_context,
                      prompt_template,
                      ip_address,
                      temperature,
                      max_length,
                      top_k,
                      top_p)

logging.basicConfig(level=logging.INFO)

db = faissDB.load(index_name)

while True:
    # // TODO, do not look for embedding for certain question,
    # // for example, if you ask give a summary of we discussed, the embedding will have nothing to do with the question

    # // TODO, embedding the history chat as well
    # // for example, a continuous question, like previous one is what is 1+1? next one is what if add 1 more?

    question = input("Your question: ")
    question = question.strip()
    if question.startswith("NEW:"):
        new_chat = True
        question = question[4:].strip()
        logging.info(f"Starting a new conversation...")
    else:
        new_chat = False
    question_embedding = embedding_function(question)
    similar_docs = [db.id2content[hid] for hid in db.query([question_embedding], top_k_match)[1][0]]
    context_text = create_context(similar_docs)
    prompt = prompt_template.format(question=question, context=context_text)

    # prepare the payload
    data = {"data": prompt,
            "temperature": temperature,
            "max_length": max_length,
            "top_k": top_k,
            "top_p": top_p,
            "session_id": "xin",
            "new_chat": new_chat,
            "system_message": system_message}

    # send request to your Flask app
    response = requests.post(f"http://{ip_address}:5000/predict", json=data)
    # Wrap text to 100 characters
    wrapped_text = textwrap.fill(response.json()['answer'], width=100)
    print(wrapped_text)

