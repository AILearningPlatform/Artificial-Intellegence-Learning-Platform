function displayImage(input) {
    const file = input.files[0];
    const reader = new FileReader();
  
    reader.onload = function (e) {
      const imgElement = document.getElementById("demoImage");
      imgElement.src = e.target.result;
    };
  
    if (file) {
      reader.readAsDataURL(file);
    }
  }
  