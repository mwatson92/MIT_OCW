from ibm_watson import VisualRecognitionV3, ApiException
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
def classifyImage(lang='en'):
    try:
        words = []
        with open('./nature.jpg', 'rb') as images_file:
            classes = visual_recognition.classify(
                images_file=images_file,
                threshold='0.6',
                accept_language=lang).get_result()
            #print(json.dumps(classes, indent=2))
            for n in range(len(classes["images"][0]["classifiers"][0]["classes"])):
                words.append(classes["images"][0]["classifiers"][0]["classes"][n]["class"])
            print(words)
    except ApiException as ex:
        print("Method failed with status code ", ex.code, ": ", ex.message)

classifyImage('en')
classifyImage('ja')
