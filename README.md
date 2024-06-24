idea:

Using nlp = 

ON images - 

generate a dataset of images with text
use ocr to identify text and bbox
use Named entity recognition model to identify pii
gaussian blur on boxes identified as pii
output same image but with pii masked

ON normal text -
generate a dataset of pdfs, the most common form of forms, extract text and location
use Named entity recognition model to identify pii 
mask with X replacing words containing pii 
output same pdf but only with pii masked
