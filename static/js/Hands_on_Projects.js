const searchInput = document.getElementById('searchInput');
const suggestionsList = document.getElementById('suggestions');

const suggestions = [
    'ResNet-50 (Image Classification)',
    'YOLOv11 (Object Detection)',
    'Mask R-CNN (Instance Segmentation)',
    'DeepLabV3+ (Semantic Segmentation)',
    'CycleGAN (Image-to-Image Translation)',
    'Gemini (Natural Language Processing)',
    'BERT (Language Understanding)',
    'MobileNetV2 (Lightweight Image Classification)',
    'Faster R-CNN (Object Detection)',
    'U-Net (Medical Image Segmentation)',
    'VGG-16 (Image Classification)',
    'OpenPose (Human Pose Estimation)',
    'Stable Diffusion (Image Generation)',
    'EfficientNet-B0 (Efficient Image Classification)',
    'CLIP (Text-to-Image Understanding)',
    'StyleGAN3 (High-Resolution Image Generation)',
    'DALL·E 2 (Creative Image Synthesis)',
    'BigGAN (High-Quality Image Generation)',
    'Vision Transformer (ViT - Transformer-based Image Classification)'
];

const MAX_SUGGESTIONS = 10;  

searchInput.addEventListener('input', () => {
    const searchTerm = searchInput.value.toLowerCase();
    suggestionsList.innerHTML = '';

    if (searchTerm.length > 0) {
        const filteredSuggestions = suggestions
            .filter(suggestion => suggestion.toLowerCase().includes(searchTerm))
            .slice(0, MAX_SUGGESTIONS);  

        filteredSuggestions.forEach(suggestion => {
            const li = document.createElement('li');
            li.textContent = suggestion;
            li.style.cursor = 'pointer';

            li.addEventListener('click', () => {
                searchInput.value = suggestion;  
            });

            suggestionsList.appendChild(li);
        });
    }
});

searchInput.addEventListener('focus', () => {
    suggestionsList.innerHTML = '';
    suggestions
        .slice(0, MAX_SUGGESTIONS)  
        .forEach(suggestion => {
            const li = document.createElement('li');
            li.textContent = suggestion;
            li.style.cursor = 'pointer';

            li.addEventListener('click', () => {
                searchInput.value = suggestion;  
            });

            suggestionsList.appendChild(li);
        });
});

searchInput.addEventListener('blur', () => {
    setTimeout(() => suggestionsList.innerHTML = '', 200);  
});

function transformText(text) {
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    text = text.replace(/```(.*?)```/gs, function(match, code) {
        const language = code.split("\n")[0]; 
        const codeContent = code.replace(language + "\n", ''); 
        return `<div class="code-block">
                  <div class="code-header">
                      <span class="code-lang">${language}</span>
                  </div>
                  <pre><code class="${language}">${codeContent}</code></pre>
                </div>`;
    });

    return text;
}

const socket = new WebSocket("ws://localhost:8000/ws/chat");
const chatbox = document.getElementById("chatbox");

socket.onopen = () => {
    console.log("WebSocket connection established.");
};

function smoothTypingEffect(container, message, speed = 50) {
    let i = 0;
    const interval = setInterval(() => {
        container.innerHTML = "Bot: " + message.substring(0, i + 1);
        i++;
        if (i >= message.length) {
            clearInterval(interval); 
        }
    }, speed);
}

function showTypingIndicator() {
    const typingContainer = document.createElement('div');
    typingContainer.classList.add('bot', 'message', 'bot-typing');
    typingContainer.innerHTML = "Bot: Typing...";
    chatbox.appendChild(typingContainer);
    chatbox.scrollTop = chatbox.scrollHeight;
    return typingContainer;
}

function removeTypingIndicator(typingContainer) {
    typingContainer.remove();
}

function sendMessage() {
    const userInput = document.getElementById("userInput");
    const text = userInput.value.trim();

    if (text !== '') {
        const userMessageContainer = document.createElement('div');
        userMessageContainer.classList.add('user', 'message');
        userMessageContainer.innerHTML = "You: " + transformText(text);
        chatbox.appendChild(userMessageContainer);

        const typingIndicator = showTypingIndicator();

        socket.send(text);
        userInput.value = ""; 
        chatbox.scrollTop = chatbox.scrollHeight; 
    }
}

socket.onmessage = (event) => {
    const message = event.data;
    if (message === "") {
        return;
    }
    const botMessageContainer = document.createElement('div');
    botMessageContainer.classList.add('bot', 'message');
    
    const transformedMessage = transformText(message);
    
    smoothTypingEffect(botMessageContainer, transformedMessage);
    
    chatbox.appendChild(botMessageContainer);
    chatbox.scrollTop = chatbox.scrollHeight; 

    const typingIndicator = document.querySelector('.bot-typing');
    if (typingIndicator) {
        typingIndicator.remove();
    }
};

document.getElementById("userInput").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});

window.onload = function() {
    const messages = document.getElementsByClassName('message');
    for(let msg of messages) {
        msg.innerHTML = transformText(msg.innerHTML);
    }
};
