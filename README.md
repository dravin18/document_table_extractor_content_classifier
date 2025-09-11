This pipeline consists of the following compoenents: 
1. Gray scaling of the input.
2. The gray scaled algorithm is sent to the table detector algorithm.
3. this algorithm returs a boumding nox and probability of the bouding box area being a table.
4. The grayscale image is cropped to sent to the oprical character recongiyion algoeirthm
5. THE ocr resutns counding box and connresponding text.
6. The cropeed image, counding boz and tezt are sent to the document llm which preidcts theclass for the text in the bounding box.
7. the bounding box ans predicted text is then drawn on the cropped image.
8. The requirements.txt has all package information to be installed.
9. Above pipline is wrapped around an api (fast api)