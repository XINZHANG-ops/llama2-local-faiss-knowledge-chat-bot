import os
import json
from datetime import datetime
from flask import Flask, render_template, session
from flask_socketio import SocketIO
from llama2_utils import llama2
from werkzeug.utils import secure_filename
import logging
import uuid
from test_vector_dbs.faissDB.faissDB import faissDB
from settings import (index_name,
                      embedding_function,
                      top_k_match,
                      system_message,
                      create_context,
                      prompt_template,
                      temperature,
                      max_length,
                      top_k,
                      top_p,
                      new_chat,
                      chat_mode,
                      embedding_hist)
logging.basicConfig(level=logging.INFO)

llama2_api = llama2('meta-llama/Llama-2-7b-chat-hf')
db = faissDB.load(index_name)
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
socketio = SocketIO(app)


def check_session(session_id):
    # Create directory "Chat_Data" if it doesn't exist
    if not os.path.exists("Chat_Data"):
        os.mkdir("Chat_Data")
    # Validate and sanitize session_id to prevent security risks
    session_id = secure_filename(session_id)
    session_path = os.path.join("Chat_Data", session_id)
    # Create directory for session_id if it doesn't exist
    if not os.path.exists(session_path):
        os.mkdir(session_path)
    return session_path


def update_chat_counter(session_id, increment=True):
    session_path = check_session(session_id)
    counter_file_path = os.path.join(session_path, "chat_counter.txt")

    if os.path.exists(counter_file_path):
        with open(counter_file_path, "r") as f:
            counter = int(f.read())
    else:
        counter = 0

    if increment:
        counter += 1

    with open(counter_file_path, "w") as f:
        f.write(str(counter))

    return counter


def data_collection(session_id, history, new):
    chat_id = update_chat_counter(session_id, new)
    session_path = check_session(session_id)

    filename = f"{chat_id}.json"
    with open(os.path.join(session_path, filename), 'w') as f:
        json.dump(history, f)

    return chat_id


def try_load_history(session_id, chat_id):
    try:
        with open(f"Chat_Data/{session_id}/{chat_id}.json", "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []
    return history


def chat_history_process(new, session_id):
    if new:
        llama2_api.reset_history()
    else:
        session_path = check_session(session_id)
        counter_file_path = os.path.join(session_path, "chat_counter.txt")
        if os.path.exists(counter_file_path):
            with open(counter_file_path, "r") as f:
                chat_id = int(f.read())
        else:
            chat_id = 0
        history = try_load_history(session_id, chat_id)
        if history:
            llama2_api.history = history
        else:
            # means it is a new session, reset history might from other session
            llama2_api.reset_history()


def run_and_update(user_prompt, temp, topp, topk, session_id, new):
    now = datetime.now()
    # format as string
    current_time = now.strftime("%Y-%m-%d %H:%M")
    logging.info(f"{current_time}@{session_id}: {user_prompt}")
    logging.info('-' * 100)

    generator = llama2_api.run(message=user_prompt,
                               temperature=temp,
                               top_p=topp,
                               top_k=topk)
    previous_texts = ""
    for response in generator:
        increment_text = response[len(previous_texts):]
        socketio.emit('new_response_increment', {'data': increment_text})
        previous_texts = response

    llama2_api.update_history('user', user_prompt)
    llama2_api.update_history('assistant', previous_texts.strip())

    time_spend = datetime.now() - now
    # format as string
    logging.info(f"Bot using {round(time_spend.total_seconds(), 3)}s reply {session_id}: {previous_texts.strip()}")
    logging.info('*' * 100)

    _ = data_collection(session_id, llama2_api.history, new)


@app.route('/')
def index():
    session['session_id'] = str(uuid.uuid4())
    session['system_message'] = system_message
    session['temperature'] = temperature
    session['max_length'] = max_length
    session['top_k'] = top_k
    session['top_p'] = top_p
    session['new_chat'] = new_chat
    session['chat_mode'] = chat_mode
    session['embedding_hist'] = embedding_hist
    return render_template('index.html')


@socketio.on('send_message')
def handle_message(payload):
    _message = payload.get("message", "")
    _sessionId = payload.get("sessionId", "")
    _systemMessage = payload.get("systemMessage", system_message)

    _temperature = float(payload.get("temperature", temperature))  # ensure it's a float
    _maxLength = int(payload.get("maxLength", max_length))  # ensure it's an integer
    _topK = int(payload.get("topK", top_k))  # ensure it's an integer
    _topP = float(payload.get("topP", top_p))  # ensure it's a float

    _new_chat = bool(payload.get("newChat", new_chat))  # ensure it's a boolean
    _chat_mode = bool(payload.get("chatMode", chat_mode))  # ensure it's a boolean
    _embedding_hist = int(payload.get("embeddingHist", embedding_hist))  # ensure it's an integer

    llama2_api.DEFAULT_SYSTEM_PROMPT = _systemMessage
    llama2_api.MAX_NEW_TOKENS = _maxLength

    user_prompt = _message.strip()
    if _chat_mode:
        chat_history_process(_new_chat, _sessionId)
        run_and_update(user_prompt, _temperature, _topP, _topK, _sessionId, _new_chat)

    # local knowledge mode
    else:
        chat_history_process(_new_chat, _sessionId)
        if len(llama2_api.history) > 0 and _embedding_hist > 0:
            hist_prompts = '\n'.join([text for role, text in llama2_api.history[-_embedding_hist:]])
            user_prompt = hist_prompts + '\n' + user_prompt

        prompt_embedding = embedding_function(user_prompt)
        similar_docs = [db.id2content[hid] for hid in db.query([prompt_embedding], top_k_match)[1][0]]
        context_text = create_context(similar_docs)
        user_prompt = prompt_template.format(question=user_prompt, context=context_text)

        run_and_update(user_prompt, _temperature, _topP, _topK, _sessionId, _new_chat)


if __name__ == '__main__':
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)








# fake_data = "Great! How can I assist you today? Do you have any questions or topics you'd like to discuss?".split(" ")
#
#
# @socketio.on('send_message')
# def handle_message(payload):
#     generator = fake_data
#
#     previous_texts = ""
#     for response in generator:
#         socketio.sleep(0.1)  # Simulate time delay
#         #logging.info(previous_texts + response)
#         increment_text = response + ' '
#         socketio.emit('new_response_increment', {'data': increment_text})
#         previous_texts += response + ' '
#
#




