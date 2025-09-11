from models import table_detector
from models import optical_character_recognizer
from models import document_entity_predictor
import cv2
import time
from fastapi import FastAPI, HTTPException, status, UploadFile
from fastapi.responses import Response

app = FastAPI()

@app.post("/get_predicted_image",response_class=Response)
async def get_predicted_image(pydantic_model: UploadFile):
    file = await pydantic_model.read()
    with open(f"/Users/user/Projects/Orbit/document_classification_app/image_store/input_image.png","wb") as f:
        f.write(file)

    start = time.time()
    cropped_image = table_detector.table_detector_pipeline(image_path="/Users/user/Projects/Orbit/document_classification_app/image_store/input_image.png",
                                        image_processor_model_path='/Users/user/Projects/Orbit/document_classification_app/models_files/table_detector/image_processor',
                                        object_detector_model_path='/Users/user/Projects/Orbit/document_classification_app/models_files/table_detector/object_detector',)

    optical_character_recognizer_results = optical_character_recognizer.optical_character_recognizer_inference(cropped_image=cropped_image)

    resulting_image = document_entity_predictor.document_entity_predictor_pipeline(optical_character_recognizer_results=optical_character_recognizer_results,
                                                                cropped_image=cropped_image,tokenizer_model_file_path='/Users/user/Projects/Orbit/document_classification_app/models_files/document_entity_predictor/tokenizer',
                                                                token_classifer_model_file_path='/Users/user/Projects/Orbit/document_classification_app/models_files/document_entity_predictor/token_classifier',
                                                                words_labels=document_entity_predictor.words_lables)
    end = time.time()
    cv2.imwrite('/Users/user/Projects/Orbit/document_classification_app/image_store/output_image.png',resulting_image)
    with open('/Users/user/Projects/Orbit/document_classification_app/image_store/output_image.png', 'rb') as image_file:
        file = image_file.read()
        try:
            return Response(
        content=file)
        except Exception as e :
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail = f"{e}")  