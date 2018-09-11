import boto3
import base64
import json
import numpy as np
import logging
import traceback,sys
from chalice import Chalice

app = Chalice(app_name='image-classification')
app.debug = True

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

@app.route('/classification', methods=['POST'], content_types=['application/octet-stream'], cors=True)
def classification():

    try:
        body_data = app.current_request.raw_body
        body_data = body_data.split(b'base64,')

        image = base64.b64decode(body_data[1])

        sagemaker_client = boto3.client(service_name='sagemaker-runtime', region_name='us-east-1')

        logger.info('Invoke Endpoint')
        res = sagemaker_client.invoke_endpoint(
                        EndpointName='<<YOUR SAGEMAKER ENDPOINT NAME>>',
                        ContentType='application/x-image',
                        Body = image
                    )

        result = res['Body'].read()
        result = json.loads(result)

        # the result will output the probabilities for all classes
        object_categories = ['ak47', 'american-flag', 'backpack', 'baseball-bat', 'baseball-glove', 'basketball-hoop', 'bat', 'bathtub', 'bear', 'beer-mug', 'billiards', 'binoculars', 'birdbath', 'blimp', 'bonsai-101', 'boom-box', 'bowling-ball', 'bowling-pin', 'boxing-glove', 'brain-101', 'breadmaker', 'buddha-101', 'bulldozer', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'cannon', 'canoe', 'car-tire', 'cartman', 'cd', 'centipede', 'cereal-box', 'chandelier-101', 'chess-board', 'chimp', 'chopsticks', 'cockroach', 'coffee-mug', 'coffin', 'coin', 'comet', 'computer-keyboard', 'computer-monitor', 'computer-mouse', 'conch', 'cormorant', 'covered-wagon', 'cowboy-hat', 'crab-101', 'desk-globe', 'diamond-ring', 'dice', 'dog', 'dolphin-101', 'doorknob', 'drinking-straw', 'duck', 'dumb-bell', 'eiffel-tower', 'electric-guitar-101', 'elephant-101', 'elk', 'ewer-101', 'eyeglasses', 'fern', 'fighter-jet', 'fire-extinguisher', 'fire-hydrant', 'fire-truck', 'fireworks', 'flashlight', 'floppy-disk', 'football-helmet', 'french-horn', 'fried-egg', 'frisbee', 'frog', 'frying-pan', 'galaxy', 'gas-pump', 'giraffe', 'goat', 'golden-gate-bridge', 'goldfish', 'golf-ball', 'goose', 'gorilla', 'grand-piano-101', 'grapes', 'grasshopper', 'guitar-pick', 'hamburger', 'hammock', 'harmonica', 'harp', 'harpsichord', 'hawksbill-101', 'head-phones', 'helicopter-101', 'hibiscus', 'homer-simpson', 'horse', 'horseshoe-crab', 'hot-air-balloon', 'hot-dog', 'hot-tub', 'hourglass', 'house-fly', 'human-skeleton', 'hummingbird', 'ibis-101', 'ice-cream-cone', 'iguana', 'ipod', 'iris', 'jesus-christ', 'joy-stick', 'kangaroo-101', 'kayak', 'ketch-101', 'killer-whale', 'knife', 'ladder', 'laptop-101', 'lathe', 'leopards-101', 'license-plate', 'lightbulb', 'light-house', 'lightning', 'llama-101', 'mailbox', 'mandolin', 'mars', 'mattress', 'megaphone', 'menorah-101', 'microscope', 'microwave', 'minaret', 'minotaur', 'motorbikes-101', 'mountain-bike', 'mushroom', 'mussels', 'necktie', 'octopus', 'ostrich', 'owl', 'palm-pilot', 'palm-tree', 'paperclip', 'paper-shredder', 'pci-card', 'penguin', 'people', 'pez-dispenser', 'photocopier', 'picnic-table', 'playing-card', 'porcupine', 'pram', 'praying-mantis', 'pyramid', 'raccoon', 'radio-telescope', 'rainbow', 'refrigerator', 'revolver-101', 'rifle', 'rotary-phone', 'roulette-wheel', 'saddle', 'saturn', 'school-bus', 'scorpion-101', 'screwdriver', 'segway', 'self-propelled-lawn-mower', 'sextant', 'sheet-music', 'skateboard', 'skunk', 'skyscraper', 'smokestack', 'snail', 'snake', 'sneaker', 'snowmobile', 'soccer-ball', 'socks', 'soda-can', 'spaghetti', 'speed-boat', 'spider', 'spoon', 'stained-glass', 'starfish-101', 'steering-wheel', 'stirrups', 'sunflower-101', 'superman', 'sushi', 'swan', 'swiss-army-knife', 'sword', 'syringe', 'tambourine', 'teapot', 'teddy-bear', 'teepee', 'telephone-box', 'tennis-ball', 'tennis-court', 'tennis-racket', 'theodolite', 'toaster', 'tomato', 'tombstone', 'top-hat', 'touring-bike', 'tower-pisa', 'traffic-light', 'treadmill', 'triceratops', 'tricycle', 'trilobite-101', 'tripod', 't-shirt', 'tuning-fork', 'tweezer', 'umbrella-101', 'unicorn', 'vcr', 'video-projector', 'washing-machine', 'watch-101', 'waterfall', 'watermelon', 'welding-mask', 'wheelbarrow', 'windmill', 'wine-bottle', 'xylophone', 'yarmulke', 'yo-yo', 'zebra', 'airplanes-101', 'car-side-101', 'faces-easy-101', 'greyhound', 'tennis-shoes', 'toad', 'clutter']

        out = ''
        index = np.argsort(result)
        for i in index[::-1]:
            out += '{} / [probability] {:.2%},'.format(object_categories[i], result[i])
            if result[i] < 0.1:
                break

        return out[:-1]

    except Exception as e:
        tb = sys.exc_info()[2]
        return 'error:{0}'.format(e.with_traceback(tb))

@app.route('/rekognition', methods=['POST'], content_types=['application/octet-stream'], cors=True)
def rekognition():

    try:
        body_data = app.current_request.raw_body
        body_data = body_data.split(b'base64,')

        image = base64.b64decode(body_data[1])

        rekognition_client = boto3.client(service_name='rekognition', region_name='us-east-1')

        logger.info('Invoke Rekognition')
        res = rekognition_client.detect_labels(
                        Image = { 'Bytes': image },
                        MaxLabels=5,
                        MinConfidence=10
                    )

        translate_client = boto3.client(service_name='translate', region_name='us-east-1')
        out = ''
        for label in res['Labels'] :
            trans = translate_client.translate_text(Text=label['Name'], 
                        SourceLanguageCode='en', TargetLanguageCode='ja')

            out += '[en] {} / [ja] {} / [Confidence] {:.2f}%,'.format(
                        label['Name'], trans.get('TranslatedText'), label['Confidence']
                    )

        return out[:-1]

    except Exception as e:
        tb = sys.exc_info()[2]
        return 'error:{0}'.format(e.with_traceback(tb))