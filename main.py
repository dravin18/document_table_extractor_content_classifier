import api
import uvicorn

if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

# from models import table_detector
# from models import optical_character_recognizer
# from models import document_entity_predictor
# import cv2
# import time
# from fastapi import FastAPI, HTTPException, status, Depends
# import schemas

# @app.post("/get_predicted_image")
# def get_predicted_image(pydantic_model: schemas.image_input_scheme):
#     file = pydantic_model.binary_image
#     with open(f"/Users/user/Projects/Orbit/document_classification_app/image_store/input_image.png","wb") as f:
#         f.write(file)

#     start = time.time()
#     cropped_image = table_detector.table_detector_pipeline(image_path="/Users/user/Projects/Orbit/document_classification_app/image_store/input_image.png",
#                                         image_processor_model_path='/Users/user/Projects/Orbit/document_classification_app/models_files/table_detector/image_processor',
#                                         object_detector_model_path='/Users/user/Projects/Orbit/document_classification_app/models_files/table_detector/object_detector',)

#     optical_character_recognizer_results = optical_character_recognizer.optical_character_recognizer_inference(cropped_image=cropped_image)

#     resulting_image = document_entity_predictor.document_entity_predictor_pipeline(optical_character_recognizer_results=optical_character_recognizer_results,
#                                                                 cropped_image=cropped_image,tokenizer_model_file_path='/Users/user/Projects/Orbit/document_classification_app/models_files/document_entity_predictor/tokenizer',
#                                                                 token_classifer_model_file_path='/Users/user/Projects/Orbit/document_classification_app/models_files/document_entity_predictor/token_classifier',
#                                                                 words_labels=document_entity_predictor.words_lables)
#     cv2.imwrite('/Users/user/Projects/Orbit/document_classification_app/image_store/output_image.png')
#     with open('/Users/user/Projects/Orbit/document_classification_app/image_store/output_image.png', 'rb') as image_file:
#         try:
#             return {'output_binary_image':image_file,'pipeline_running_time':end-start}
#         except:
#             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail = f"There is an error in the code")
#         ]

# end = time.time()
# print("PIPELINE_RUNNING_TIME:",end-start)
# cv2.imshow('output_image',resulting_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('output_2.png',resulting_image)