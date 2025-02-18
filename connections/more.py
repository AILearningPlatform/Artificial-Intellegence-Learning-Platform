from connections.check import *

models = {
    "yolo11": YOLO("connections/models_or_datasets/yolo11n.pt"),
    "yolov8": YOLO("connections/models_or_datasets/yolov8s.pt"),
    "resnet50_image_classification": torch.load(os.path.join(MODEL_FOLDER, "resnet50.pt")),
    "mask_rcnn_instance_segmentation": torch.load(os.path.join(MODEL_FOLDER, "mask_rcnn_resnet50_fpn.pt")),
    "vgg16": torch.load(os.path.join(MODEL_FOLDER, "vgg16.pt")),
    "faster_rcnn": torch.load(os.path.join(MODEL_FOLDER, "faster_rcnn.pt")),
    "mobilenet_v2": torch.load(os.path.join(MODEL_FOLDER, "mobilenet_v2.pt")),
}

 
class Models:

    @staticmethod
    def Mask_R_CNN_Instance_Segmentation(image_path):
        total_preds = []
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        classes = weights.meta['categories']
        model = models['mask_rcnn_instance_segmentation']
        model.eval()
        preprocess = T.Compose([T.ToTensor()])
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            predictions = model(img_tensor)

        pred_masks = predictions[0]['masks']
        pred_labels = predictions[0]['labels']
        pred_scores = predictions[0]['scores']
        pred_boxes = predictions[0]['boxes']
        threshold = 0.5

        valid_preds = [
            (mask, label.item(), score.item(), box)
            for mask, label, score, box in zip(pred_masks, pred_labels, pred_scores, pred_boxes)
            if score.item() > threshold
        ]

        img_np = np.array(img)  
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(img_np)

        for mask, label, score, box in valid_preds:
            total_preds.append(classes[label])  

            mask = mask[0].cpu().numpy()   
            mask = np.where(mask > 0.5, 1, 0)  

            ax.imshow(mask, cmap='jet', alpha=0.6)

            xmin, ymin, xmax, ymax = box.cpu().numpy()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            ax.text(xmin, ymin, f'{classes[label]} ({score:.2f})', color='red', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

        plt.axis('off')  
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  

        plt.savefig("static/saved/segmented_image.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig) 

        return [f"Image: {image_path[18:]} \nPredicted objects: {total_preds}", "static/saved/segmented_image.png"]


    @staticmethod
    def ResNet_50_Image_Classification(image_path):
        model = models['resnet50_image_classification']
        model.eval()
        
        preprocess = ResNet50_Weights.DEFAULT.transforms()
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
        
        _, top_class = outputs.max(1)
        
        class_labels = ResNet50_Weights.DEFAULT.meta["categories"]
        predicted_label = class_labels[top_class.item()]
        
        #show(img, title=f"{predicted_label}", sz=5)

        return [f"Image: {image_path[18:]} {predicted_label}", image_path]


    @staticmethod
    def CycleGAN_Image_to_Image_Translation(image_path):
        return [f"Image: {image_path[18:]} Predicted: Gwapo", image_path]

    @staticmethod
    def YOLOv11_Object_Detection(image_path):
        saved_path = f"static/saved/YOLO11_obj.png"
        img = read(image_path)
        try:

            pred = models['yolo11'](img)

            boxes = pred[0].boxes
            labels = boxes.cls  

            all_preds = [(pred[0].names[labels[i].item()], boxes.xyxy[i].cpu().numpy().tolist()) for i in range(len(labels))]

            class_counts = Counter([name for name, _ in all_preds])

            if class_counts:
                result = "Predictions: " + ", ".join([f"{count} {name}s" for name, count in class_counts.items()])
            else:
                result = "No predictions."

            show(img, sz = 5, bbs=boxes.xyxy.cpu().numpy(), title=result)
            
            return [f"Image: {image_path[18:]} prediction: {result}", image_path]
        
        except ValueError:
            result = "No predictions"
            return [f"Image: {image_path[18:]} prediction: {result}", image_path]

    @staticmethod
    def Faster_R_CNN_Object_Detectio(image_path):
        new_loc = r"static/saved/Faster R-CNN (Object Detection).png"
        model = models['faster_rcnn']
        model.eval()
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        classes = weights.meta['categories']
        img = cv2.imread(image_path)  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_pil = Image.fromarray(img)
        img_tensor = F.to_tensor(img_pil).unsqueeze(0)  
        with torch.no_grad():
            prediction = model(img_tensor)
        boxes, labels, scores = prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']
        threshold = 0.5
        high_confidence_indices = scores > threshold
        boxes = boxes[high_confidence_indices]
        labels = labels[high_confidence_indices]
        scores = scores[high_confidence_indices]
        for box in boxes:
            x1, y1, x2, y2 = box.tolist() 
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(new_loc, bbox_inches='tight', pad_inches=0)
        return [f"Image: {image_path[18:]} Predicted: {str([classes[label] for label in labels] if len(labels) >= 1 else 'No Predictions')[1:-1]}", "/static/saved/Faster R-CNN (Object Detection).png"]


    @staticmethod
    def VGG_16_Image_Classification(image_path):
        try:
            model = models['vgg16']
            weights = VGG16_Weights.DEFAULT
            classes = weights.meta['categories']

            for param in model.parameters():
                param.requires_grad = False
            model.eval()

            trnsfrms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            img = Image.open(image_path).convert("RGB")
            input_tensor = trnsfrms(img).unsqueeze(0) 

            with torch.no_grad():
                output = model(input_tensor)
                logits = output[0]  

            z = torch.argmax(logits)  
            p = logits[z].item()  

            show(img, sz=8, title=f"{classes[z]} (logit: {p:.2f})")

            predicted_label = classes[z]
            return [f"Image: {image_path.split('/')[-1]} {predicted_label}", image_path]

        except ValueError as e:
            result = "No predictions"
            return [f"Image: {image_path.split('/')[-1]} {result}. Error: {str(e)}", image_path]
        

    @staticmethod
    def MobileNetV2(image_path):
        class_idx_path = "connections/models_or_datasets/imagenet_class_index.json"
        model = models['mobilenet_v2']


        preprocess = T.Compose([
            T.Resize(256),               
            T.CenterCrop(224),        
            T.ToTensor(),                
            T.Normalize(               
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        with open(class_idx_path, "r") as f:
            class_idx = json.load(f)

        class_labels = [class_idx[str(i)][1] for i in range(1000)]


        img = Image.open(image_path).convert("RGB") 
        processed_img = preprocess(img)  

        try:
            model.eval()  
            with torch.no_grad():
                processed_img = processed_img.unsqueeze(0)   
                output = model(processed_img)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)  
                top_prob, top_idx = probabilities.topk(1)  
                predicted_label = class_labels[top_idx.item()]  

                print(f"Predicted Label: {predicted_label}")
                print(f"Probability: {top_prob.item():.4f}")
        except Exception as e:
            print(f"Error running model inference: {e}")
    
        return [f"Image: {image_path[18:]} Prediction: {predicted_label} {100 * top_prob.item():.2f} %", image_path]


    @staticmethod
    def Deep_seekR1(prompt):
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
        res = model(prompt)
        return res 