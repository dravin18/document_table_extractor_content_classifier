import cv2
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch

def image_grayscaling(image_path: str):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def table_detector_inference(processed_image,image_processor_model_path,object_detector_model_path):
    image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    image_processor = AutoImageProcessor.from_pretrained(image_processor_model_path)
    model = TableTransformerForObjectDetection.from_pretrained(object_detector_model_path)
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # update id2label to include "no object"
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"

    return outputs, id2label

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_h, img_w = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def table_detector_results_post_processing(outputs,img_size,id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [int(elem) for elem in bbox]})

    return objects

def image_cropper(objects,processed_image):
    y_start = objects[0]['bbox'][1]
    y_end = objects[0]['bbox'][3]
    x_start = objects[0]['bbox'][0]
    x_end = objects[0]['bbox'][2]
    cropped_image = processed_image[y_start:y_end,x_start:x_end]
    return cropped_image

def table_detector_pipeline(image_path,image_processor_model_path,object_detector_model_path):
    processed_image = image_grayscaling(image_path)
    outputs, id2label = table_detector_inference(processed_image,image_processor_model_path,object_detector_model_path)
    objects = table_detector_results_post_processing(outputs,processed_image.shape,id2label)
    cropped_image = image_cropper(objects,processed_image)
    return cropped_image