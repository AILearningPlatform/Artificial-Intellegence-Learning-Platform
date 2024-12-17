from connections.check import *

models = {
    "yolo11": YOLO("static/models_or_datasets/yolo11n.pt"),
    "yolov8": YOLO("static/models_or_datasets/yolov8n.pt"),
    "resnet50_ImaGE_Classification": resnet50(weights=None),
    "Mask_R_CNN_Instance_Segmentation": maskrcnn_resnet50_fpn(weights = "DEFAULT")
}

class Models:

    @staticmethod
    def Mask_R_CNN_Instance_Segmentation(image_path):
        total_preds = []
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        classes = weights.meta['categories']
        model = models['Mask_R_CNN_Instance_Segmentation']
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

            ax.imshow(mask, cmap='jet', alpha=0.4)

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
        model_loc = "static/models_or_datasets/resnet_50.pth"

        model = torch.load(model_loc, map_location=torch.device('cpu'))
        model.eval()

        preprocess = ResNet50_Weights.DEFAULT.transforms()
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_prob, top_class = probabilities.topk(1, dim=1)

        class_labels = ResNet50_Weights.DEFAULT.meta["categories"]
        predicted_label = class_labels[top_class.item()]

        show(img, title=f"{predicted_label} ({top_prob.item() * 100:.2f}%)", sz = 5)

        return [f"Image: {image_path[18:]} {predicted_label} ({top_prob.item() * 100:.2f}%)", image_path]


    @staticmethod
    def CycleGAN_Image_to_Image_Translation(image_path):
        return [f"Image: {image_path[18:]} Predicted: Gwapo", image_path]

    @staticmethod
    def YOLOv11_Object_Detection(image_path):
        path = image_path
        img = read(path)
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

            return [f"Image: {image_path[18:]} {result}", image_path]
        
        except ValueError:
            result = "No predictions"
            return [f"Image: {image_path[18:]} {result}", image_path]

