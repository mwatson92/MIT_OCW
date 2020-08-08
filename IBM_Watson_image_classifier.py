from ibm_watson import ApiException
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json

# Authentication
APIKEY = "EmTJFtnzXWVTtYwl86QSjwMiMPk_LyOJwrgyHjalCcaz"
VERSION = "2018-03-19"
URL = "https://api.us-south.visual-recognition.watson.cloud.ibm.com/instances/8a836b63-cb82-49b4-8696-404e07088a5c"

authenticator = IAMAuthenticator(APIKEY)
visual_recognition = VisualRecognitionV3(
    version=VERSION,
    authenticator=authenticator
)

visual_recognition.set_service_url(URL)

# Classify an image
try:
    with open('./IMG_0136_copy.jpeg', 'rb') as images_file:
        classes = visual_recognition.classify(
            images_file=images_file,
            threshold='0.6',
            accept_language='en').get_result()
        print(json.dumps(classes, indent=2))
except ApiException as ex:
    print("Method failed with status code ", ex.code, ": ", ex.message)
