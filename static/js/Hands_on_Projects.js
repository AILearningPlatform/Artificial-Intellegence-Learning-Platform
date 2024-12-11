const searchInput = document.getElementById('searchInput');
const suggestionsList = document.getElementById('suggestions');

const suggestions = [
    'ResNet-50 (Image Classification)',
    'YOLOv11 (Object Detection)',
    'Mask R-CNN (Instance Segmentation)',
    'DeepLabV3+ (Semantic Segmentation)',
    'CycleGAN (Image-to-Image Translation)',
    'GPT-4 (Natural Language Processing)',
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
