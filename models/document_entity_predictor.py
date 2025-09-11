from transformers import LayoutLMv2Processor
from transformers import LayoutLMv2ForTokenClassification
import cv2
import random


def getting_bounding_box_and_text(optical_character_recognizer_results):
    words = []
    boxes = []
    for ind_result in optical_character_recognizer_results:
        words.append(ind_result[1])
        box_top_left = ind_result[0][0]
        print(box_top_left)
        box_bottom_right = ind_result[0][2]
        print(box_bottom_right)
        box = box_top_left + box_bottom_right
        boxes.append(box)
    return words,boxes


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


def generating_dummy_word_labels(words):
    dummy_labels_list = []
    for _ in range(0,len(words),1):
        dummy_labels_list.append(random.randrange(0,6,1))
    return dummy_labels_list


def tokenizer_input_processing(words,boxes,cropped_image,tokenizer_model_file_path,dummy_labels):
    height, width = cropped_image.shape
    normalize_boxes = []
    for box in boxes:
        normalize_boxes.append(normalize_bbox(box,width,height))
    cropped_rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
    processor = LayoutLMv2Processor.from_pretrained(tokenizer_model_file_path,revision="no_ocr")

    encoded_inputs = processor(cropped_rgb_image,words, boxes=normalize_boxes,word_labels=dummy_labels,
                            padding="max_length", truncation=True, return_tensors="pt")
    labels = encoded_inputs.pop('labels').squeeze().tolist()
    return encoded_inputs, labels, cropped_rgb_image


def token_classifier_inference(encoded_inputs,token_classifer_model_file_path):
    # load the fine-tuned model from the hub
    model = LayoutLMv2ForTokenClassification.from_pretrained(token_classifer_model_file_path)
    # forward pass
    outputs = model(**encoded_inputs)
    return outputs


def unnormalize_box(bbox, width, height):
     return [
         int(width * (bbox[0] / 1000)),
         int(height * (bbox[1] / 1000)),
         int(width * (bbox[2] / 1000)),
         int(height * (bbox[3] / 1000)),
     ]


def iob_to_label(label):
    label = label[2:]
    if not label:
      return 'other'
    return label


words_lables = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']


def token_classifier_results_post_processing(cropped_image,encoded_inputs,outputs,words_labels,labels):
    token_boxes = encoded_inputs.bbox.squeeze().tolist()
    height, width, _ = cropped_image.shape
    true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    id2label = {v: k for v, k in enumerate(words_labels)}
    predictions = [id2label[prediction] for prediction, label in zip(predictions, labels) if label != -100]
    label2color = {'question': (0,0,255), 'answer':(0, 255, 0), 'header':(255, 165, 0), 'other':(255, 0, 255)}
    for prediction, box in zip(predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        cropped_image = cv2.rectangle(cropped_image,(box[0],box[1]),(box[2],box[3]),
                                      label2color[predicted_label],2)
        cropped_image = cv2.putText(cropped_image, predicted_label,(box[0] + 10, box[1] - 10),cv2.FONT_HERSHEY_SIMPLEX,
                                    1,label2color[predicted_label],2,cv2.LINE_AA)
    return cropped_image


def document_entity_predictor_pipeline(optical_character_recognizer_results,cropped_image,tokenizer_model_file_path,token_classifer_model_file_path,words_labels):
    words, boxes = getting_bounding_box_and_text(optical_character_recognizer_results)
    dummy_labels = generating_dummy_word_labels(words)
    encoded_inputs, labels, cropped_rgb_image = tokenizer_input_processing(words,boxes,cropped_image,tokenizer_model_file_path,dummy_labels)
    outputs = token_classifier_inference(encoded_inputs,token_classifer_model_file_path)
    words_predicted_image = token_classifier_results_post_processing(cropped_rgb_image,encoded_inputs,outputs,words_labels,labels)
    return words_predicted_image