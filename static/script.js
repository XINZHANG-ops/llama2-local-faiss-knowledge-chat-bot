var socket = io.connect('http://' + document.domain + ':' + location.port);
var fullText = "";

socket.on('connect', function() {
    socket.emit('send_message', {data: 'I\'m connected!'});
});

socket.on('new_response_increment', function(data) {
    fullText += data.data;
    var p = document.createElement('p');
    p.innerHTML = `<b>Assistant:</b> ${fullText}`;
    p.className = "assistant-font";

    var chatbox = document.getElementById('chatbox');
    while (chatbox.firstChild) {
        chatbox.removeChild(chatbox.firstChild);
    }
    chatbox.appendChild(p);

    // Scroll to the bottom
    chatbox.scrollTop = chatbox.scrollHeight;
});

document.getElementById("submit_button").addEventListener("click", function() {
    var message = document.getElementById("message").value;
    socket.emit('send_message', {data: message});
});
