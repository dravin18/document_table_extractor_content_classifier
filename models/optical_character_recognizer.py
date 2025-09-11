import easyocr

def optical_character_recognizer_inference(cropped_image):
    reader = easyocr.Reader(['en'])
    optical_character_recognizer_results = reader.readtext(cropped_image)
    return optical_character_recognizer_results