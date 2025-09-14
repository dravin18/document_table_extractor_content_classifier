from models import table_detector
from models import optical_character_recognizer
from models import document_entity_predictor
import cv2
import time
import os
from fastapi import FastAPI, HTTPException, status, UploadFile
from fastapi.responses import Response

app = FastAPI()

@app.post("/get_predicted_image",response_class=Response)
async def get_predicted_image(pydantic_model: UploadFile):
    file = await pydantic_model.read()
    current_cwd = os.getcwd()
    input_image_directory = os.path.join(current_cwd, 'image_store', 'input_image.png')
    image_processor_model_dir = os.path.join(current_cwd, 'model_files', 'table_detector', 'image_processor')
    object_detector_model_dir = os.path.join(current_cwd, 'model_files', 'table_detector', 'object_detector')
    tokenizer_path = os.path.join(current_cwd, 'model_files', 'document_entity_predictor', 'tokenizer')
    token_classifier_path = os.path.join(current_cwd, 'model_files', 'document_entity_predictor', 'token_classifier')
    output_image_directory = os.path.join(current_cwd, 'image_store')

    with open(input_image_directory,"wb") as f:
        f.write(file)

    start = time.time()
    cropped_image = table_detector.table_detector_pipeline(image_path = input_image_directory,
                                        image_processor_model_path = image_processor_model_dir,
                                        object_detector_model_path = object_detector_model_dir,)

    optical_character_recognizer_results = optical_character_recognizer.optical_character_recognizer_inference(cropped_image=cropped_image)

    resulting_image = document_entity_predictor.document_entity_predictor_pipeline(optical_character_recognizer_results=optical_character_recognizer_results,
                                                                cropped_image=cropped_image,tokenizer_model_file_path=tokenizer_path,
                                                                token_classifer_model_file_path=token_classifier_path,
                                                                words_labels=document_entity_predictor.words_lables)
    end = time.time()
    cv2.imwrite(output_image_directory,resulting_image)
    with open(os.path.join(output_image_directory, 'output_image.png'), 'rb') as image_file:
        file = image_file.read()
        try:
            return Response(content=file)
        except Exception as e :
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail = f"{e}")  